from datasets import load_from_disk, concatenate_datasets
from global_vars import MAX_SHARD_SIZE
import pathlib
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--nb_workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    dataset_names = args.datasets.split(',')

    for i, dataset_name in enumerate(dataset_names):
        print(f"Loading {dataset_name}...")
        raw_data = load_from_disk(input_dir / dataset_name)

        if i == 0:
            all_data = raw_data
        else:
            all_data = concatenate_datasets([all_data, raw_data])

        print(f"{dataset_name} has been added to the final dataset.")

    all_data = all_data.shuffle(seed=args.seed)

    output_dir = input_dir

    all_data.save_to_disk(
        output_dir / f"all_training_data", max_shard_size=MAX_SHARD_SIZE, num_proc=args.nb_workers
    )


if __name__ == "__main__":
    main()
