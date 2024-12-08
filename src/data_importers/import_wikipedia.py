from datasets import load_dataset, concatenate_datasets
import argparse
from global_vars import DATA_DIR, MAX_SHARD_SIZE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepped_by_hg_file", action=argparse.BooleanOptionalAction)
    parser.add_argument("--language", type=str, default=None, required=False)
    parser.add_argument("--date", type=str, default=None, required=True) # e.g. 20220301
    args = parser.parse_args()


    if args.prepped_by_hg_file:
        # AR: Not sure I actually trust this version, it seems quite small (200MB)
        # and doesn't have language splits.
        dataset = load_dataset(
            "wikipedia",
            f"{args.date}.simple"
        )
    else:
        dataset = load_dataset(
            "wikipedia",
            date=args.date,
            use_auth_token=True,
            language=args.language,
            beam_runner="DirectRunner"
        )

    dataset = concatenate_datasets(list(dataset.values()))

    dataset.save_to_disk(
        DATA_DIR / "raw" / f"Wikipedia_{args.date}_{args.language.upper()}",
        max_shard_size=MAX_SHARD_SIZE,
    )



if __name__ == "__main__":
    main()
