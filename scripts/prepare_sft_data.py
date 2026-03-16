"""
Prepare SFT data for Qwen3 training, aligned with RL rollout / eval.

Splits each multi-step trajectory into per-step samples:
  - prompt: conversation history rendered by chat template (historical thinking stripped)
  - response: current step's assistant response WITH <think> block

This matches what the model sees during RL rollout / eval exactly:
  - History has no <think> (template strips it, same as inference)
  - Current step generates <think>reasoning</think> followed by action

Requires tokenizer access (run on the training server, not locally).

Usage:
    python scripts/prepare_sft_data.py \
        --input data/swe/R2EGym_SFT_Trajectories.parquet \
        --output data/swe/R2EGym_SFT_Trajectories_Qwen3.parquet \
        --model_path /path/to/Qwen3-8B
"""

import argparse
import re

import pandas as pd


def extract_thinking_and_action(content: str) -> tuple[str, str]:
    """
    Split assistant message into (reasoning, action).

    Input:  "Let me search.\n\n<function=search>\n  <parameter=term>foo</parameter>\n</function>"
    Output: ("Let me search.", "<function=search>\n  <parameter=term>foo</parameter>\n</function>")
    """
    # Already has <think> tags
    if "<think>" in content and "</think>" in content:
        think_start = content.index("<think>") + len("<think>")
        think_end = content.index("</think>")
        reasoning = content[think_start:think_end].strip("\n")
        action = content[think_end + len("</think>"):].lstrip("\n")
        return reasoning, action

    # Find first tool call
    tool_call_match = re.search(r"<function=\w+>", content)
    if tool_call_match:
        pre_tool = content[: tool_call_match.start()].rstrip("\n")
        tool_and_rest = content[tool_call_match.start():]
        return pre_tool, tool_and_rest
    else:
        # Final message (no tool call)
        return content.strip(), ""


def split_trajectory(messages: list[dict]) -> list[tuple[list[dict], str]]:
    """
    Split a trajectory into per-step (context_messages, completion) pairs.

    Returns list of (messages_for_prompt, completion_string) tuples.
    Historical assistant turns have thinking stripped.
    """
    samples = []
    context = []

    for msg in messages:
        if msg["role"] in ("system", "user"):
            context.append(msg)

        elif msg["role"] == "assistant":
            reasoning, action = extract_thinking_and_action(msg["content"])

            # Build completion with <think> for this step
            if reasoning:
                completion = f"<think>\n{reasoning}\n</think>\n\n{action}"
            else:
                completion = f"<think>\n</think>\n\n{action}"

            # Save sample: current context → completion
            samples.append((list(context), completion))

            # Add this turn to context WITHOUT thinking (matches inference)
            context.append({"role": "assistant", "content": action})

    return samples


def render_prompts(samples, tokenizer):
    """Render message lists into prompt strings using chat template."""
    rendered = []
    for context_messages, completion in samples:
        prompt = tokenizer.apply_chat_template(
            context_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        rendered.append({"prompt": prompt, "response": completion})
    return rendered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/swe/R2EGym_SFT_Trajectories.parquet")
    parser.add_argument("--output", default="data/swe/R2EGym_SFT_Trajectories_Qwen3.parquet")
    parser.add_argument("--model_path", required=True, help="Path to Qwen3 model (for tokenizer)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"Loaded tokenizer from {args.model_path}")

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} trajectories from {args.input}")

    # Split trajectories into per-step samples
    all_samples = []
    for idx, row in df.iterrows():
        messages = list(row["messages"])
        samples = split_trajectory(messages)
        all_samples.extend(samples)

    print(f"Split into {len(all_samples)} per-step samples "
          f"(avg {len(all_samples)/len(df):.1f} steps/trajectory)")

    # Render prompts using chat template
    rendered = render_prompts(all_samples, tokenizer)
    out_df = pd.DataFrame(rendered)

    out_df.to_parquet(args.output, index=False)
    print(f"\nSaved {len(out_df)} samples to {args.output}")


if __name__ == "__main__":
    main()
