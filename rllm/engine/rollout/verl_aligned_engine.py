"""
VeRL-aligned rollout engine for local backends (sglang / vllm).

Replicates VerlEngine.get_model_response() logic — token-ID prompt,
tokenizer.decode(skip_special_tokens=True), chat_parser.parse_completion —
but calls the server via OpenAI-compatible /v1/completions HTTP endpoint
instead of verl's Ray-based AsyncLLMServerManager.
"""

import asyncio
import logging

import openai

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)

# Parameters accepted directly by the openai SDK's completions.create().
# Non-standard params (e.g. top_k for vLLM/SGLang) must go through extra_body.
_OPENAI_COMPLETION_PARAMS = frozenset(
    {
        "best_of",
        "echo",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_tokens",
        "n",
        "presence_penalty",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "suffix",
        "temperature",
        "top_p",
        "user",
    }
)


def _split_extra_body(params: dict) -> None:
    """Move non-standard params into extra_body for vLLM/SGLang compatibility."""
    extra = {k: params.pop(k) for k in list(params) if k not in _OPENAI_COMPLETION_PARAMS}
    if extra:
        params["extra_body"] = {**params.get("extra_body", {}), **extra}


class VerlAlignedEngine(RolloutEngine):
    """Drop-in rollout engine that mirrors VerlEngine's encode/decode logic."""

    def __init__(
        self,
        model: str,
        tokenizer,
        base_url: str,
        api_key: str = "EMPTY",
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        sampling_params: dict | None = None,
        api_retries: int = 3,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.chat_parser = ChatTemplateParser.get_parser(
            tokenizer,
            disable_thinking=kwargs.get("disable_thinking", False),
        )
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.sampling_params = sampling_params or {}
        self.api_retries = api_retries
        self.accumulate_reasoning = kwargs.get("accumulate_reasoning", False)
        self.reasoning_effort = self.sampling_params.pop("reasoning_effort", "medium")
        self.tools = kwargs.get("tools", [])

        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """VeRL-aligned: token IDs in, tokenizer.decode(skip_special_tokens=True) out."""
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)
        kwargs.pop("model", None)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        tools = kwargs.pop("tools", self.tools)
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
        reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        max_tokens = sampling_params.pop(
            "max_tokens",
            sampling_params.pop("max_new_tokens", self.max_response_length),
        )

        # ── 1. Build prompt string (same as VerlEngine) ──
        prompt = self.chat_parser.parse(
            messages,
            add_generation_prompt=True,
            is_first_msg=True,
            tools=tools,
            accumulate_reasoning=accumulate_reasoning,
            reasoning_effort=reasoning_effort,
        )

        # ── 2. Encode to token IDs (same as VerlEngine) ──
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_ids)

        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        # ── 3. Call /v1/completions with token IDs ──
        _split_extra_body(sampling_params)

        retries = self.api_retries
        while retries > 0:
            try:
                response = await self.client.completions.create(
                    model=self.model,
                    prompt=prompt_ids,
                    max_tokens=max_tokens,
                    timeout=3600,
                    **sampling_params,
                )

                # ── 4. Extract completion IDs ──
                try:
                    completion_ids = response.choices[0].token_ids
                    assert completion_ids is not None
                except Exception:
                    raw_text = response.choices[0].text
                    completion_ids = self.tokenizer.encode(
                        raw_text,
                        add_special_tokens=False,
                    )

                # ── 5. Enforce max_tokens (same as VerlEngine) ──
                finish_reason = response.choices[0].finish_reason
                if len(completion_ids) >= max_tokens:
                    finish_reason = "length"
                    completion_ids = completion_ids[:max_tokens]

                # ── 6. Decode with skip_special_tokens=True (matches VerlEngine) ──
                completion_text = self.tokenizer.decode(
                    completion_ids,
                    skip_special_tokens=True,
                )

                # ── 7. Parse completion (same as VerlEngine) ──
                parsed_output = self.chat_parser.parse_completion(completion_ids)

                return ModelOutput(
                    text=completion_text,
                    content=parsed_output["content"],
                    reasoning=parsed_output["reasoning"],
                    tool_calls=parsed_output["tool_calls"],
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    logprobs=[],
                    prompt_logprobs=[],
                    prompt_length=response.usage.prompt_tokens,
                    completion_length=response.usage.completion_tokens,
                    finish_reason=finish_reason,
                )

            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    raise Exception("Rate limit reached and retries exhausted.") from None
                print("Sleep for 5 seconds for API limit.")
                await asyncio.sleep(5)

            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise Exception(f"Error processing content after retries: {e}") from e
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(1)
