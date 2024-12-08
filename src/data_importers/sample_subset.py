import os
from datasets import load_from_disk, concatenate_datasets
from global_vars import MAX_SHARD_SIZE
import pathlib
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--nb_workers", type=int, default=10)
    parser.add_argument("--n_perc", type=int, default=10)
    args = parser.parse_args()
    
    random.seed(42)

    input_dir = pathlib.Path(args.input_dir)
    dataset_names = args.datasets.split(',')

    for i, dataset_name in enumerate(dataset_names):
        print(f"Loading {dataset_name}...")
        raw_data = load_from_disk(input_dir / dataset_name)
        print("Removing columns...")
        cols_to_remove = [k for k in raw_data.column_names if k != 'text']
        clean_data = raw_data.map(lambda x: x, remove_columns=cols_to_remove, num_proc = args.nb_workers)
        indices = random.sample(list(range(len(clean_data))), int(len(clean_data) * args.n_perc / 100))
        print("Selecting indices...")
        clean_sub_data = clean_data.select(indices)
        
        if i == 0:
            all_data_perc = clean_sub_data
        else:
            all_data_perc = concatenate_datasets([all_data_perc, clean_sub_data])

        print(f"Preprocessing and sampling for {dataset_name} done.")
    
    output_dir = input_dir

    all_data_perc.save_to_disk(
            output_dir / f"all_data_{args.n_perc}perc", max_shard_size=MAX_SHARD_SIZE, num_proc=args.nb_workers
        )


if __name__ == "__main__":
    main()
