from datasets import load_from_disk
from tqdm import tqdm
import pathlib
import argparse

def count_tokens(dataset):
    total_tokens = 0
    for example in tqdm(dataset):
        total_tokens += len(example["input_ids"])
    return total_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    dataset = load_from_disk(input_dir / args.dataset)

    total_tokens = count_tokens(dataset)

    print(f"Total tokens in the dataset: {total_tokens}")

if __name__ == "__main__":
    main()
