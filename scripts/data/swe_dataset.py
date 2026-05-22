import argparse
import json
import os

import pandas as pd
from datasets import load_dataset
from verl.utils.hdfs_io import copy, makedirs

import rllm
from rllm.agents.system_prompts import SWE_SYSTEM_PROMPT, SWE_USER_PROMPT

# Get the directory for rLLM repo (rllm.__file__)
RLLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(rllm.__file__)))

SWE_DATASETS = [
    "R2E-Gym/R2E-Gym-Subset",
    # "R2E-Gym/R2E-Gym-Lite",
    # "R2E-Gym/R2E-Gym-V1",
    # "R2E-Gym/SWE-Bench-Lite",
    "R2E-Gym/SWE-Bench-Verified",
    # "r2e-edits/SweSmith-RL-Dataset",
    "R2E-Gym/R2EGym-SFT-Trajectories"
]

HARD_DIFFICULTIES = {"1-4 hours", ">4 hours"}


def _get_hard_instance_ids():
    """Get instance_ids of hard problems from SWE-bench/SWE-bench_Verified (difficulty labels)."""
    print("Downloading SWE-bench/SWE-bench_Verified for difficulty labels ...")
    ds = load_dataset("SWE-bench/SWE-bench_Verified")
    split = ds["test"] if "test" in ds else ds["train"]
    hard_ids = set()
    for row in split:
        if row.get("difficulty") in HARD_DIFFICULTIES:
            hard_ids.add(row["instance_id"])
    print(f"Found {len(hard_ids)} hard instances from SWE-bench/SWE-bench_Verified")
    return hard_ids


def main():
    parser = argparse.ArgumentParser(description="Generate trajectories using specified environment and policy.")
    parser.add_argument("--local_dir", default=os.path.join(RLLM_DIR, "data/swe"))
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    hdfs_dir = args.hdfs_dir

    def make_map_fn():
        def process_fn(row):
            row_dict = dict(row)
            problem_statement = row_dict.get("problem_statement", "")
            return {
                "data_source": "swe",
                "prompt": [{"role": "system", "content": SWE_SYSTEM_PROMPT}, {"role": "user", "content": SWE_USER_PROMPT.format(problem_statement=problem_statement)}],
                "ability": "swe",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": json.dumps(row_dict),
            }

        return process_fn

    process_fn = make_map_fn()

    for dataset_name in SWE_DATASETS:
        print(f"Processing dataset: {dataset_name}")
        try:
            # Load the dataset dictionary (which contains splits like 'train' or 'test')
            dataset_splits = load_dataset(dataset_name)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue

        output_name_base = dataset_name.split("/")[-1].replace("-", "_")  # Use underscore for consistency

        # Determine which split exists ('train' or 'test')
        if "train" in dataset_splits:
            split_name = "train"
            split_data = dataset_splits["train"]
        elif "test" in dataset_splits:
            split_name = "test"
            split_data = dataset_splits["test"]
        else:
            print(f"Skipping {dataset_name} as it contains neither 'train' nor 'test' split.")
            continue

        print(f"Using '{split_name}' split for {dataset_name}")

        # SFT trajectory data: save raw without reformatting
        if "trajectories" in dataset_name.lower():
            df = split_data.to_pandas()
            output_filepath = os.path.join(local_dir, f"{output_name_base}.parquet")
            df.to_parquet(output_filepath)
            print(f"Saved {len(df)} SFT records from '{split_name}' split to {output_filepath} (no reformat)")
            continue

        # Process the data from the identified split
        processed_data = [process_fn(row) for row in split_data]

        # Create DataFrame and save to a single parquet file
        df = pd.DataFrame(processed_data)
        output_filepath = os.path.join(local_dir, f"{output_name_base}.parquet")
        df.to_parquet(output_filepath)
        print(f"Saved {len(df)} records from '{split_name}' split to {output_filepath}")

        # Copy to HDFS if needed
        if hdfs_dir is not None:
            hdfs_filepath = os.path.join(hdfs_dir, f"{output_name_base}.parquet")
            # Ensure HDFS directory exists before copying
            # Assuming makedirs handles potential race conditions or existing dirs
            try:
                makedirs(hdfs_dir)
                copy(src=output_filepath, dst=hdfs_filepath)
                print(f"Copied {output_name_base}.parquet to HDFS: {hdfs_filepath}")
            except Exception as e:
                print(f"Failed to copy {output_filepath} to HDFS {hdfs_filepath}: {e}")

    # Build hard subset from SWE-Bench-Verified
    build_hard_subset(local_dir, process_fn, hdfs_dir)


def build_hard_subset(local_dir, process_fn, hdfs_dir=None):
    """Build SWE-Bench-Verified hard subset using difficulty labels from the original dataset."""
    hard_output = os.path.join(local_dir, "SWE_Bench_Verified_Hard.parquet")
    if os.path.exists(hard_output):
        print(f"{hard_output} already exists, skipping.")
        return

    # Load R2E-Gym version (has docker_image field needed for ARL pods)
    print("Building SWE-Bench-Verified hard subset ...")
    dataset_splits = load_dataset("R2E-Gym/SWE-Bench-Verified")
    split_data = dataset_splits["test"] if "test" in dataset_splits else dataset_splits["train"]

    hard_ids = _get_hard_instance_ids()
    hard_rows = [process_fn(row) for row in split_data if row.get("instance_id") in hard_ids]

    df = pd.DataFrame(hard_rows)
    df.to_parquet(hard_output)
    print(f"Saved {len(df)} hard instances to {hard_output}")

    if hdfs_dir is not None:
        try:
            makedirs(hdfs_dir)
            copy(src=hard_output, dst=os.path.join(hdfs_dir, "SWE_Bench_Verified_Hard.parquet"))
        except Exception as e:
            print(f"Failed to copy hard subset to HDFS: {e}")


if __name__ == "__main__":
    main()
