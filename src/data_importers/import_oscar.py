from datasets import load_dataset, concatenate_datasets
import pathlib
import argparse
import json
from global_vars import DATA_DIR, MAX_SHARD_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--nb_workers", type=int, default=1, required=False)
    args = parser.parse_args()

    dataset = load_dataset(
        "oscar-corpus/OSCAR-2201",
        use_auth_token=True,
        language=args.language,
        streaming=False,
        num_proc=int(args.nb_workers)
    )
    dataset = concatenate_datasets(list(dataset.values()))

    dataset.save_to_disk(
        DATA_DIR / "raw" / f"OSCAR_2201_{args.language.upper()}",
        max_shard_size=MAX_SHARD_SIZE,
    )


if __name__ == "__main__":
    main()
