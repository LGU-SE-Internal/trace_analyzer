"""
Prepare SFT data for Qwen3 training by wrapping reasoning text in <think> tags.

Problem:
    Qwen3's chat template uses `loop.last` logic to inject think blocks:
    - In full conversation: only the LAST assistant turn (after last_query_index) gets think block
    - In MultiTurnSFTDataset partial applications: EVERY assistant turn is "last",
      so ALL turns get empty <think>\n\n</think>\n\n injected
    - full_tokens != concat_tokens → validation warning → fallback to concat_tokens
    - Model trains to produce empty think blocks for every response

Fix:
    Pre-process assistant messages to include real reasoning in <think> tags.
    The explanatory text before a tool call becomes the <think> content.
    After this, concat_tokens has real thinking content, which is what we want.

Usage:
    python scripts/prepare_sft_data.py \
        --input data/swe/R2EGym_SFT_Trajectories.parquet \
        --output data/swe/R2EGym_SFT_Trajectories_Qwen3.parquet
"""

import argparse
import re

import pandas as pd


def wrap_thinking(content: str) -> str:
    """
    Wrap assistant message content so reasoning goes in <think> tags.

    Before: "Let me look at the code.\n\n<function=search>..."
    After:  "<think>\nLet me look at the code.\n</think>\n\n<function=search>..."

    For final turns (no tool call):
    Before: "Done! The fix is complete."
    After:  "<think>\nDone! The fix is complete.\n</think>\n\n"
    """
    # Already has <think> tags — leave as-is
    if "<think>" in content:
        return content

    # Find the first tool call
    tool_call_match = re.search(r"<function=\w+>", content)

    if tool_call_match:
        pre_tool = content[: tool_call_match.start()].rstrip("\n")
        tool_and_rest = content[tool_call_match.start() :]

        if pre_tool:
            # Has reasoning before the tool call — wrap it
            return f"<think>\n{pre_tool}\n</think>\n\n{tool_and_rest}"
        else:
            # Bare tool call with no preceding reasoning
            return f"<think>\n</think>\n\n{tool_and_rest}"
    else:
        # No tool call — this is a final summary/completion message
        # Wrap everything as the model's final thinking
        return f"<think>\n{content.strip()}\n</think>\n\n"


def process_messages(messages: list[dict]) -> list[dict]:
    """Process a conversation's messages, wrapping assistant content."""
    result = []
    for msg in messages:
        if msg.get("role") == "assistant":
            new_msg = dict(msg)
            new_msg["content"] = wrap_thinking(msg["content"])
            result.append(new_msg)
        else:
            result.append(msg)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/swe/R2EGym_SFT_Trajectories.parquet")
    parser.add_argument("--output", default="data/swe/R2EGym_SFT_Trajectories_Qwen3.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # Process messages column
    df["messages"] = df["messages"].apply(lambda msgs: process_messages(list(msgs)))

    # Verify sample
    sample_msgs = df.iloc[0]["messages"]
    assistant_turns = [m for m in sample_msgs if m.get("role") == "assistant"]
    print(f"\nSample: {len(sample_msgs)} total messages, {len(assistant_turns)} assistant turns")
    for i, turn in enumerate(assistant_turns[:3]):
        content_preview = turn["content"][:100].replace("\n", "\\n")
        has_think = "<think>" in turn["content"]
        empty_think = "<think>\n\n</think>" in turn["content"] or "<think>\n</think>" in turn["content"]
        print(f"  Turn {i}: has_think={has_think}, empty={empty_think}, preview={content_preview!r}")

    df.to_parquet(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
