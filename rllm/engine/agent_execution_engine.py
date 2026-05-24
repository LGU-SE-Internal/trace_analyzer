import asyncio
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch

from rllm.agents.agent import Action, BaseAgent, Trajectory
from rllm.agents.utils import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.env_utils import (
    compute_mc_return,
    compute_trajectory_reward,
)
from rllm.parser import ChatTemplateParser
from rllm.utils import colorful_print

logger = logging.getLogger(__name__)


def _check_think_format(response: str) -> bool:
    """Check that all <think> tags are properly closed (paired)."""
    open_count = len(re.findall(r"<think>", response))
    close_count = len(re.findall(r"</think>", response))
    pair_count = len(re.findall(r"<think>(.*?)</think>", response, re.DOTALL))
    return open_count == close_count == pair_count


def _check_xml_tool_format(response: str) -> bool:
    """Check that XML tool calls (<function=...>...</function>) are well-formed.

    Returns True if the response contains a valid tool call or no tool call
    attempt at all (the agent might just be thinking). Returns False only when
    the response *attempts* a tool call but the XML is malformed (unclosed tags,
    missing function name, etc.).
    """
    open_count = len(re.findall(r"<function\s*=", response))
    close_count = len(re.findall(r"</function>", response))
    if open_count == 0 and close_count == 0:
        # No tool call attempt — valid (agent might be producing thought only).
        return True
    if open_count != close_count:
        return False
    # Every <function=NAME> block should be parseable.
    pairs = re.findall(r"<function\s*=\s*([^>]+)>.*?</function>", response, re.DOTALL)
    return len(pairs) == open_count


def check_trajectory_format(episode_steps: list[dict], use_fn_calling: bool = False) -> bool:
    """Validate format across all steps in a trajectory.

    Checks:
    1. <think> tag closure on every assistant response.
    2. Tool call format (XML) on every assistant response (skipped for fn_calling
       mode since the API enforces JSON schema).

    Returns False if any step violates the format.
    """
    for step in episode_steps:
        response = step.get("response", "")
        if not response:
            continue
        # Check <think> closure
        if not _check_think_format(response):
            return False
        # Check tool format (only for XML mode)
        if not use_fn_calling and not _check_xml_tool_format(response):
            return False
    return True


class AgentExecutionEngine:
    def __init__(
        self,
        engine_name="openai",
        tokenizer=None,
        rollout_engine=None,
        chat_parser=None,
        n_parallel_agents=128,  # The number of active agents
        trajectory_timeout=None,
        gamma=0.2,
        api_retries=3,
        retry_limit=3,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        config=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=64,  # The number of concurrent env operations
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if rollout_engine_args is None:
            rollout_engine_args = {}
        if env_args is None:
            env_args = {}

        self.config = config
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.max_env_workers = max_workers
        self.overlong_filter = overlong_filter

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length
        self.disable_thinking = self.config.get("rllm", {}).get("disable_thinking", False) if self.config is not None else False

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args

        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]

        self.trajectory_timeout = trajectory_timeout
        if not trajectory_timeout:
            self.trajectory_timeout = int(1e9)

        if env_class is not None:
            assert env_class.is_multithread_safe(), "Environment must be multithread safe for async engine"

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=self.disable_thinking)
        else:
            self.chat_parser = chat_parser

        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", {})  # for openai api requests

        assert self.engine_name in ["openai", "verl", "tinker"], "Currently only openai, verl and tinker are supported as rollout engine"
        if self.engine_name == "openai":
            from rllm.engine.rollout.openai_engine import OpenAIEngine

            self.rollout_engine = OpenAIEngine(
                **rollout_engine_args,
                api_retries=api_retries,
                tokenizer=self.tokenizer,
                max_prompt_length=self.max_prompt_length,
                max_response_length=self.max_response_length,
                disable_thinking=self.disable_thinking,
            )
        elif self.engine_name == "verl":
            from rllm.engine.rollout.verl_engine import VerlEngine

            self.rollout_engine = VerlEngine(
                config=self.config,
                rollout_manager=rollout_engine,
                tokenizer=self.tokenizer,
                disable_thinking=self.disable_thinking,
            )
        elif self.engine_name == "tinker":
            from rllm.engine.rollout.tinker_engine import TinkerEngine

            self.rollout_engine = TinkerEngine(
                **rollout_engine_args,
            )

        # Create a thread pool executor for environment interactions (i.e. step, reset, close)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def get_model_response(self, prompt, application_id, **kwargs) -> str:
        """
        Compute model response asynchronously based on the engine type.

        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.

        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            **kwargs: Additional arguments to pass to the model

        Returns:
            The model's response text

        Raises:
            NotImplementedError: If the engine type is not supported
        """

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        if self.engine_name == "openai":
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, enforce_max_prompt_length=False, **sampling_params)
            return output
        elif self.engine_name == "verl":
            meta_data = sampling_params.pop("meta_info", {})
            validate = meta_data.get("validate", False)
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, validate=validate, enforce_max_prompt_length=False, **sampling_params)
            return output
        elif self.engine_name == "tinker":
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, enforce_max_prompt_length=False, **sampling_params)
            return output
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def update_envs_and_agents(self, envs, agents):
        """
        Update the environments and agents.

        Args:
            envs: List of environments to use
            agents: List of agents to use
        """
        assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
        self.envs = envs
        # For keeping track of the environment index in the batch.
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run a single agent's trajectory asynchronously"""
        # Token-in-token-out mode: bypass the standard flow entirely
        tito_enabled = self.config.get("rllm", {}).get("token_in_token_out", False) if self.config is not None else False
        if tito_enabled and mode == "Token" and self.engine_name == "verl":
            return await self._run_trajectory_tito(idx, application_id, seed=seed, **kwargs)

        agent = self.agents[idx]
        env = self.envs[idx]
        # env_id = env.env_id

        termination_reason = None
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        # for step return
        episode_steps = []

        # Reset environment with the task using the executor
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        # Reset agent
        agent.reset()
        # Update agent internal state from environment.
        agent.update_from_env(
            observation=observation,  # Raw observation from environment
            reward=0.0,
            done=False,
            info=info,
        )
        messages = agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=True, contains_generation_msg=True)
        prompt_token_len = len(prompt_tokens)
        # Note, this should never happen!
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(f"Trajectory {idx}: initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length}, retrying")

        for step_idx in range(self.max_steps):
            # Get action from agent
            prompt_messages = agent.chat_completions.copy()
            # Max remaining tokens left for the response
            # For enforced max prompt at each step, no need to deduct here
            if not self.enforce_max_prompt_length:
                max_tokens = self.max_response_length - response_token_len
            else:
                max_tokens = self.max_response_length

                # since max prompt is enforced, we filter out too long prompts.
                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

            kwargs["max_tokens"] = max_tokens

            start_time = time.time()
            model_output = await self.get_model_response(prompt_messages, application_id, **kwargs)
            response = model_output.text
            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time
            # Update steps
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response,
                "prompt_ids": model_output.prompt_ids,
                "completion_ids": model_output.completion_ids,
                "logprobs": model_output.logprobs,
            }
            episode_steps.append(prompt_response_pair)

            # Update agent with model response
            action: Action = agent.update_from_model(response)
            action = action.action

            # Take step in environment using the executor
            start_time = time.time()

            try:
                next_observation, reward, done, info = await asyncio.wait_for(loop.run_in_executor(self.executor, env.step, action), timeout=(self.trajectory_timeout - total_time))
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n", "red")
                reward = 0

                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            delta_time = time.time() - start_time
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len

            # Update agent internal state.
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            # Check and convert to tokens if necessary
            assert assistant_message is not None or mode != "Token", "Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assert env_messages is not None or mode != "Token", "Environment messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assistant_msg_tokens, assistant_msg_masks = [], []
            env_msg_tokens, env_msg_masks = [], []
            if assistant_message:
                assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks([assistant_message], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=False)
            if env_messages:
                env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=True)

            # Update repsonse token length
            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
            # Reached maximum number of tokens for the trajectory
            if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
                # Truncation length
                truncation_length = self.max_response_length - response_token_len
                # Truncate the response and masks
                if truncation_length < 0:
                    truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                    truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                else:
                    # Edge case where the response is exactly the max response length.
                    truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
                    truncated_response_masks = assistant_msg_masks + env_msg_masks
                # Update token collections
                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)

                cur_step = agent.get_current_state()
                if response_token_len - len(env_msg_tokens) > self.max_response_length:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                # handle returning
                break

            # Update the token version of trajectory
            response_tokens.extend(assistant_msg_tokens)
            response_masks.extend(assistant_msg_masks)
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            # Check if episode is done
            if done:
                termination_reason = "ENV_DONE"
                break

            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        masked_out = False
        if self.overlong_filter:
            if termination_reason in ("TRUNCATION", "MAX_STEPS", "TIMEOUT", "ENV_TIMEOUT"):
                # Mask out the entire response for incomplete trajectories.
                response_masks = [0] * len(response_masks)
                masked_out = True

        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step.reward = reward

        # Format reward: penalise malformed <think> tags or tool call XML.
        # Skip format check during validation — val reward should reflect pure task performance.
        is_validate = kwargs.get("meta_info", {}).get("validate", False)
        format_ok = True
        if not masked_out and episode_steps and not is_validate:
            use_fn_calling = getattr(agent, "use_fn_calling", False)
            format_ok = check_trajectory_format(episode_steps, use_fn_calling=use_fn_calling)
            if not format_ok:
                reward = 0.0
                cur_step = agent.get_current_state()
                cur_step.reward = 0.0
                response_masks = [0] * len(response_masks)
                masked_out = True
                colorful_print(
                    f"Trajectory {idx} format violation detected — reward forced to 0, masks zeroed.\n",
                    "red",
                )

        # Closing environment using the executor.
        await loop.run_in_executor(self.executor, env.close)
        if termination_reason:
            if reward > 0:
                color = "green"
            else:
                color = "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
                color,
            )
            if masked_out:
                colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "red")

        trajectory: Trajectory = agent.trajectory
        # Aggregate final trajectory statistics
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)
        trajectory.info["termination_reason"] = termination_reason
        trajectory.info["metrics"] = {
            "steps": len(trajectory.steps),
            "reward_time": reward_time,
            "env_time": env_time,
            "llm_time": llm_time,
            "total_time": total_time,
            "format_ok": format_ok,
        }

        if mode == "Text":
            return trajectory
        elif mode == "Token":
            prompt_tokens, response_tokens, response_masks, is_valid_trajectory = self.assemble_steps(episode_steps)
            token_result = {
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "response_masks": response_masks,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "chat_completions": agent.chat_completions,
                "metrics": {
                    # Total number of steps taken in the trajectory
                    "steps": len(trajectory.steps),
                    # Time to calculate reward
                    "reward_time": reward_time,
                    # Total time spent in environment execution (env.step)
                    "env_time": env_time,
                    # Time to calculate response tokens
                    "llm_time": llm_time,
                    # Total time spent in the trajectory
                    "total_time": total_time,
                    "token_mismatch": 0.0 if is_valid_trajectory else 1.0,
                },
            }
            return token_result
        elif mode == "Conversation":
            return agent.chat_completions
        elif mode == "Step":
            steps_result = {
                "steps": episode_steps,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
                "termination_reason": termination_reason,
            }
            return steps_result
        else:
            raise ValueError(f"Mode {mode} not supported")

    def assemble_steps(self, steps: list[dict]):
        """
        Transform step-by-step results into trajectory format for training.
        The assemble is aggresive, if steps is not cumulative, the response_masks is set to all 0s.

        Each step_result contains:
        - steps: List of {"prompt": str, "response": str, "prompt_ids": list, "completion_ids": list}

        For training, we need to assemble the full conversation sequence where:
        - prompt_tokens: Initial prompt (first step's prompt_ids)
        - response_tokens: All subsequent conversation (completion_ids + next step's prompt_ids)
        - response_masks: Mask indicating which tokens contribute to loss (only completion_ids)
        """

        # Start with initial prompt from first step
        initial_prompt_ids = steps[0]["prompt_ids"]
        accumulated_sequence = initial_prompt_ids.copy()
        response_tokens = []
        response_masks = []
        is_valid_trajectory = True

        for i, step in enumerate(steps):
            current_prompt_ids = step["prompt_ids"]
            current_completion_ids = step["completion_ids"]

            if i == 0:
                # First step: just add completion
                response_tokens.extend(current_completion_ids)
                response_masks.extend([1] * len(current_completion_ids))  # completion contributes to loss
                accumulated_sequence.extend(current_completion_ids)
            else:
                if current_prompt_ids[: len(accumulated_sequence)] != accumulated_sequence:
                    # Find the first differing position
                    prefix = current_prompt_ids[: len(accumulated_sequence)]
                    diff_pos = None
                    for i, (expected, actual) in enumerate(zip(accumulated_sequence, prefix, strict=False)):
                        if expected != actual:
                            diff_pos = i
                            break

                    if diff_pos is not None:
                        logger.warning(f"When assemble steps, detect the trajectory not accumulative at position {diff_pos}. Expected: {accumulated_sequence[diff_pos : diff_pos + 5]}, Got: {prefix[diff_pos : diff_pos + 5]}. Setting response_masks to all 0s. This is likely due to retokenization.")
                    else:
                        logger.warning(f"When assemble steps, detect length mismatch. Expected length: {len(accumulated_sequence)}, Got length: {len(prefix)}. Setting response_masks to all 0s.")

                    is_valid_trajectory = False
                    break

                response_tokens.extend(current_prompt_ids[len(accumulated_sequence) :] + current_completion_ids)
                response_masks.extend([0] * (len(current_prompt_ids) - len(accumulated_sequence)) + [1] * len(current_completion_ids))  # completion contributes to loss
                accumulated_sequence = current_prompt_ids + current_completion_ids

        assert len(response_masks) == len(response_tokens)

        prompt_tokens = torch.tensor(initial_prompt_ids, dtype=torch.long)
        response_tokens = torch.tensor(response_tokens, dtype=torch.long)
        response_masks = torch.tensor(response_masks, dtype=torch.long)

        if self.config.rllm.filter_token_mismatch:
            response_masks = response_masks * int(is_valid_trajectory)

        return prompt_tokens, response_tokens, response_masks, is_valid_trajectory

    # ── Token-in-Token-out helpers ──────────────────────────────────────

    def _init_tito_anchors(self):
        """Pre-compute delta-encode anchors for token-in-token-out mode.

        Uses a fixed BASE_MSGS sentinel to derive the token offset contributed
        by chat-template wrapping so that we can encode a *single* new user/env
        message and extract only the delta tokens.
        """
        if hasattr(self, "_tito_anchors_ready"):
            return
        # Sentinel conversation — never sent to the model; only used as an
        # anchor so that ``encode(BASE + [new_msg])[anchor:]`` gives us the
        # tokens for *just* the new message (including its role/eot wrapper).
        BASE_MSGS = [{"role": "system", "content": "You are a helpful assistant."}]
        base_no_gen = self.chat_parser.parse(BASE_MSGS, add_generation_prompt=False, is_first_msg=True)
        base_with_gen = self.chat_parser.parse(BASE_MSGS, add_generation_prompt=True, is_first_msg=True)
        self._tito_base_msgs = BASE_MSGS
        self._tito_base_no_gen_len = len(self.tokenizer.encode(base_no_gen, add_special_tokens=False))
        self._tito_gen_prompt_ids: list[int] = self.tokenizer.encode(base_with_gen, add_special_tokens=False)[self._tito_base_no_gen_len :]
        self._tito_anchors_ready = True

    def _delta_encode_env_message(self, observation: str) -> list[int]:
        """Encode a user/env message and return only the delta token IDs.

        The delta excludes the system-prompt anchor and includes the full
        ``<|im_start|>user\\n{observation}<|im_end|>\\n`` wrapper.
        """
        msg = {"role": "user", "content": observation}
        full_str = self.chat_parser.parse(
            self._tito_base_msgs + [msg],
            add_generation_prompt=False,
            is_first_msg=True,
        )
        full_ids = self.tokenizer.encode(full_str, add_special_tokens=False)
        return full_ids[self._tito_base_no_gen_len :]

    async def _run_trajectory_tito(self, idx, application_id, seed=0, **kwargs):
        """Token-in-token-out trajectory execution.

        Maintains ``accumulated_ids`` (the full token sequence seen by the
        model) and ``response_mask`` (1 = LLM-generated, 0 = env/prompt)
        incrementally.  LLM completion_ids are appended directly — no
        decode → re-encode cycle.
        """
        agent = self.agents[idx]
        env = self.envs[idx]
        self._init_tito_anchors()

        termination_reason = None
        total_time = 0.0
        llm_time = 0.0
        env_time = 0.0
        reward_time = None
        reward = 0.0

        # for step-mode compatibility
        episode_steps = []

        loop = asyncio.get_event_loop()

        # ── Reset ──
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        agent.reset()
        agent.update_from_env(observation=observation, reward=0.0, done=False, info=info)

        # ── Initial prompt: encode once ──
        initial_messages = agent.chat_completions.copy()
        initial_prompt_str = self.chat_parser.parse(
            initial_messages,
            add_generation_prompt=True,
            is_first_msg=True,
        )
        accumulated_ids: list[int] = self.tokenizer.encode(initial_prompt_str, add_special_tokens=False)
        prompt_len = len(accumulated_ids)

        if prompt_len > self.max_prompt_length:
            agent.reset()
            raise Exception(f"Trajectory {idx}: initial prompt length {prompt_len} exceeded max_prompt_length {self.max_prompt_length}, retrying")

        # response_* track everything AFTER the initial prompt
        response_ids: list[int] = []
        response_mask: list[int] = []
        all_logprobs: list[float] = []
        response_token_len = 0

        for step_idx in range(self.max_steps):
            # ── Compute per-step max_tokens ──
            if not self.enforce_max_prompt_length:
                max_tokens = self.max_response_length - response_token_len
            else:
                max_tokens = self.max_response_length
                cur_prompt_len = len(accumulated_ids)
                if cur_prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

            if max_tokens <= 0:
                termination_reason = "TRUNCATION"
                break

            step_kwargs = dict(kwargs)
            step_kwargs["max_tokens"] = max_tokens

            # ── Generate: token-in, token-out ──
            start_time = time.time()
            validate = step_kwargs.pop("meta_info", {}).get("validate", False)
            # Remove non-sampling params that the inference engine doesn't understand
            step_kwargs.pop("tools", None)
            step_kwargs.pop("accumulate_reasoning", None)
            step_kwargs.pop("enforce_max_prompt_length", None)
            step_kwargs.pop("reasoning_effort", None)
            token_output = await self.rollout_engine.generate_from_token_ids(
                prompt_ids=accumulated_ids,
                application_id=application_id,
                validate=validate,
                **step_kwargs,
            )
            completion_ids: list[int] = token_output.token_ids
            logprobs: list[float] = token_output.log_probs

            # Enforce max_tokens
            if len(completion_ids) >= max_tokens:
                completion_ids = completion_ids[:max_tokens]
                logprobs = logprobs[:max_tokens] if logprobs else []

            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time

            # ── Decode for agent (text only, tokens stay untouched) ──
            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            # Record episode step (compatible with assemble_steps format)
            episode_steps.append(
                {
                    "prompt_ids": list(accumulated_ids),
                    "completion_ids": list(completion_ids),
                    "logprobs": logprobs,
                    "prompt": "",  # not used in tito mode
                    "response": completion_text,
                }
            )

            # ── Accumulate completion tokens (token-in-token-out core) ──
            accumulated_ids = accumulated_ids + list(completion_ids)
            response_ids.extend(completion_ids)
            response_mask.extend([1] * len(completion_ids))
            if logprobs:
                all_logprobs.extend(logprobs)
            response_token_len += len(completion_ids)

            # ── Update agent with decoded text ──
            action: Action = agent.update_from_model(completion_text)
            action = action.action

            # ── Environment step ──
            start_time = time.time()
            try:
                next_observation, reward, done, info = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, env.step, action),
                    timeout=(self.trajectory_timeout - total_time),
                )
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(
                        f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action.\n",
                        "red",
                    )
                reward = 0
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            delta_time = time.time() - start_time
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len

            # Update agent internal state
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            # ── Check truncation BEFORE appending env tokens ──
            # Delta-encode the env observation
            env_msg_ids = self._delta_encode_env_message(next_observation)
            gen_prompt_ids = self._tito_gen_prompt_ids
            env_turn_ids = env_msg_ids + gen_prompt_ids
            env_turn_len = len(env_turn_ids)

            if not self.enforce_max_prompt_length and (response_token_len + env_turn_len) >= self.max_response_length:
                cur_step = agent.get_current_state()
                cur_step.done = True
                termination_reason = "TRUNCATION"
                break

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            if done:
                termination_reason = "ENV_DONE"
                break

            # ── Accumulate env tokens (mask=0, not trained on) ──
            accumulated_ids = accumulated_ids + env_turn_ids
            response_ids.extend(env_turn_ids)
            response_mask.extend([0] * env_turn_len)
            response_token_len += env_turn_len

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        # ── Overlong filter ──
        masked_out = False
        if self.overlong_filter:
            if termination_reason in ("TRUNCATION", "MAX_STEPS", "TIMEOUT", "ENV_TIMEOUT"):
                response_mask = [0] * len(response_mask)
                masked_out = True

        # ── Final reward ──
        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step.reward = reward

        # ── Format reward: penalise malformed <think> tags or tool call XML ──
        # Skip format check during validation — val reward should reflect pure task performance.
        is_validate = kwargs.get("meta_info", {}).get("validate", False)
        format_ok = True
        if not masked_out and episode_steps and not is_validate:
            use_fn_calling = getattr(agent, "use_fn_calling", False)
            format_ok = check_trajectory_format(episode_steps, use_fn_calling=use_fn_calling)
            if not format_ok:
                reward = 0.0
                cur_step = agent.get_current_state()
                cur_step.reward = 0.0
                response_mask = [0] * len(response_mask)
                masked_out = True
                colorful_print(
                    f"Trajectory {idx} format violation detected — reward forced to 0, masks zeroed.\n",
                    "red",
                )

        # ── Close env ──
        await loop.run_in_executor(self.executor, env.close)

        if termination_reason:
            color = "green" if reward > 0 else "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}.\n",
                color,
            )
            if masked_out:
                colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "red")

        # ── Trajectory stats ──
        trajectory: Trajectory = agent.trajectory
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)
        trajectory.info["termination_reason"] = termination_reason
        trajectory.info["metrics"] = {
            "steps": len(trajectory.steps),
            "reward_time": reward_time,
            "env_time": env_time,
            "llm_time": llm_time,
            "total_time": total_time,
            "format_ok": format_ok,
        }

        # ── Build result (Token mode) ──
        prompt_tokens = torch.tensor(accumulated_ids[:prompt_len], dtype=torch.long)
        response_tokens_t = torch.tensor(response_ids, dtype=torch.long)
        response_masks_t = torch.tensor(response_mask, dtype=torch.long)

        return {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens_t,
            "response_masks": response_masks_t,
            "trajectory_reward": trajectory.reward,
            "idx": env.idx,
            "chat_completions": agent.chat_completions,
            "metrics": {
                "steps": len(trajectory.steps),
                "reward_time": reward_time,
                "env_time": env_time,
                "llm_time": llm_time,
                "total_time": total_time,
                "token_mismatch": 0.0,  # always valid in tito mode
            },
        }

    async def run_agent_trajectory_with_retry(self, idx, seed=0, mode="Text", **kwargs):
        loop = asyncio.get_event_loop()
        last_exception = None
        for attempt in range(self.retry_limit):
            try:
                application_id = str(uuid.uuid4())
                return await asyncio.wait_for(self.run_agent_trajectory_async(idx, application_id=application_id, seed=seed, mode=mode, **kwargs), timeout=7200)
            except Exception as e:
                last_exception = e
                colorful_print(f"Trajectory {idx}: attempt {attempt + 1}/{self.retry_limit} failed: {e}", "red")
                # Clean up the environment on failure before retrying
                # This ensures pods are deleted even when exceptions occur
                try:
                    env = self.envs[idx]
                    if env is not None and hasattr(env, "close"):
                        await loop.run_in_executor(self.executor, env.close)
                        colorful_print(f"Trajectory {idx}: Cleaned up environment after exception", "yellow")
                except Exception as cleanup_error:
                    colorful_print(f"Trajectory {idx}: Failed to clean up environment: {cleanup_error}", "red")
                continue
        colorful_print(f"Trajectory {idx}: all {self.retry_limit} attempts failed. Last error: {last_exception}", "red")
        raise Exception(f"Trajectory {idx} cannot complete. Please check the log message") from last_exception

    async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", **kwargs):
        if timing_raw is None:
            timing_raw = {}
        assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
        assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"  # type: ignore
        max_concurrency = self.n_parallel_agents

        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

        if self.engine_name == "verl":
            await self.rollout_engine.wake_up()  # type: ignore

        semaphore = asyncio.Semaphore(self.n_parallel_agents)

        async def launch_one_trajectory_task(env_idx: int):
            async with semaphore:
                try:
                    result = await self.run_agent_trajectory_with_retry(
                        idx=env_idx,
                        seed=reset_seed,
                        mode=mode,
                        **kwargs,
                    )
                except Exception as e:
                    colorful_print(f"Trajectory {env_idx}: all retries exhausted, returning zeroed-out dummy trajectory: {e}", "red")
                    env = self.envs[env_idx]
                    if mode == "Token":
                        # Create a minimal dummy with correct tensor types and zeroed masks
                        # so it contributes nothing to training but keeps batch size consistent.
                        dummy_prompt = torch.tensor([self.tokenizer.bos_token_id or 0], dtype=torch.long)
                        dummy_response = torch.tensor([self.tokenizer.eos_token_id or 0], dtype=torch.long)
                        result = {
                            "prompt_tokens": dummy_prompt,
                            "response_tokens": dummy_response,
                            "response_masks": torch.zeros(1, dtype=torch.long),
                            "trajectory_reward": 0.0,
                            "idx": env.idx,
                            "chat_completions": [],
                            "metrics": {
                                "steps": 0,
                                "reward_time": 0.0,
                                "env_time": 0.0,
                                "llm_time": 0.0,
                                "total_time": 0.0,
                                "token_mismatch": 1.0,
                            },
                        }
                    else:
                        raise e
                return result

        # Create all N conceptual tasks. Their execution will be throttled by the semaphore
        # and the availability of agent/env indices.
        tasks_to_run = [launch_one_trajectory_task(i) for i in range(len(self.envs))]

        tasks_completed = 0
        for coro in asyncio.as_completed(tasks_to_run):
            try:
                result = await coro
                tasks_completed += 1
                colorful_print(f"Number of Trajectories {tasks_completed}/{len(self.envs)} completed", "cyan")
                yield result
            except Exception as e:
                raise e

        if self.engine_name == "verl":
            await self.rollout_engine.sleep()  # type: ignore

        self.executor.shutdown(wait=False, cancel_futures=True)

    async def execute_tasks(self, tasks: list[dict]):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.

        Args:
            tasks: List of tasks to process
            max_concurrent: Maximum number of concurrent tasks to process (defaults to self.n_parallel_agents)

        Returns:
            A list of trajectories, one for each task.
        """
        if not hasattr(self, "executor") or self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=self.max_env_workers)

        max_concurrent = self.n_parallel_agents

        # Initialize results list to store trajectories for all tasks
        all_trajectories = {}

        # Create a queue of tasks to process
        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        # Track completed trajectories
        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with semaphore:
                # Get an available index
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                    self.agents[index] = self.agent_class(**self.agent_args)
                    assert self.agents[index] is not None and isinstance(self.agents[index], BaseAgent), "Agent is not initalized or not inheriting from BaseAgent"
                    self.agents[index].trajectory.task = task  # type: ignore
                    res = await self.run_agent_trajectory_async(index, application_id=task_id)
                    res.task = task
                    res.info["chat_completions"] = self.agents[index].chat_completions
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, res
                except Exception as exc:
                    completed += 1
                    logger.error("Task %s failed (%d/%d): %s", task_id, completed, total, exc)
                    return task_id, None
                finally:
                    # Ensure env is closed to release sandbox pods
                    env = self.envs[index] if index < len(self.envs) else None
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                        self.envs[index] = None
                    # Put the index back in the queue when done
                    await index_queue.put(index)

        # Run all tasks concurrently
        results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])

        all_trajectories = {task_id: trajectory for task_id, trajectory in results if trajectory is not None}
        failed = total - len(all_trajectories)
        if failed:
            logger.warning("%d/%d tasks failed and were excluded from results", failed, total)
        ordered_trajectories = [all_trajectories[i] for i in range(total) if i in all_trajectories]

        self.executor.shutdown(wait=False, cancel_futures=True)

        return ordered_trajectories

    def shutdown(self):
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown()
            self.executor = None


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
