import os
from datasets import load_from_disk, concatenate_datasets
from global_vars import MAX_SHARD_SIZE
import pathlib
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--nb_workers", type=int, default=10)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = input_dir

    dataset_names = os.listdir(input_dir)

    for dataset_name in dataset_names:
        print(f"Loading {dataset_name}...")
        dataset = load_from_disk(input_dir / dataset_name)
        dataset = concatenate_datasets(list(dataset.values()))

        dataset.save_to_disk(
            output_dir / dataset_name, max_shard_size=MAX_SHARD_SIZE, num_proc=args.nb_workers
        )

        print(f"Merged dataset for {dataset_name} saved.")

if __name__ == "__main__":
    main()
