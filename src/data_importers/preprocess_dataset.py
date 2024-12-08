from transformers import LlamaTokenizerFast, AutoTokenizer
import os
from datasets import load_from_disk
from global_vars import DATA_DIR, MODEL_DIR, MAX_SHARD_SIZE
import argparse
from llama_recipes_patch.datasets.utils import Concatenator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--nb_workers", type=int, default=10)
    parser.add_argument("--datasets", required=False, type=str)
    parser.add_argument("--block_size", default=4096, type=int)
    parser.add_argument('--from_raw', action=argparse.BooleanOptionalAction)
    parser.add_argument('--from_tokenized', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.from_tokenized and args.from_raw:
        raise ValueError('Either from_raw or from_tokenized should be set, but not both.')

    LM_tokenized_output_dir = DATA_DIR / "LM_tokenized" / args.tokenizer
    if LM_tokenized_output_dir.exists():
        print("LM dataset already exists, but will be processed again.")
    
    if 'llama' in args.tokenizer.lower():
        tokenizer = LlamaTokenizerFast.from_pretrained(MODEL_DIR / "tokenizer" / args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer" / args.tokenizer)

    if args.from_raw:
        input_dir = DATA_DIR / "raw"
    elif args.from_tokenized:
        input_dir = DATA_DIR / "tokenized"
    else:
        input_dir = DATA_DIR / "processed_raw"

    if args.datasets is not None:
        dataset_names = args.datasets.split(',')
    else:
        dataset_names = os.listdir(input_dir)

    for dataset_name in dataset_names:
        print(f"Loading {dataset_name}...")
        dataset = load_from_disk(input_dir / dataset_name)

        if args.from_raw:
            print(f"Preprocessing raw dataset {dataset_name}...")
            cols = dataset.column_names
            cols.remove("text")
            dataset = dataset.map(lambda x: x, remove_columns=cols, num_proc = args.nb_workers)

        if not args.from_tokenized:
            print(f"Tokenizing {dataset_name}...")
            dataset = dataset.map(
                lambda sample: tokenizer(sample["text"]),
                batched=True,
                remove_columns=list(dataset.features),
                num_proc=args.nb_workers,
            )
            #dataset.save_to_disk(
            #    tokenized_output_dir / dataset_name, max_shard_size=MAX_SHARD_SIZE, num_proc=args.nb_workers
            #)
        
        print(f"Concatenating {dataset_name}...")
        dataset = dataset.map(Concatenator(chunk_size=args.block_size), batched=True, num_proc=args.nb_workers)

        dataset.save_to_disk(
            LM_tokenized_output_dir / dataset_name, max_shard_size=MAX_SHARD_SIZE, num_proc=args.nb_workers
        )

        print(f"LM dataset for {dataset_name} saved.")

if __name__ == "__main__":
    main()
