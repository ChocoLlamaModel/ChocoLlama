from transformers import AutoModelForCausalLM
import transformers
import argparse
from global_vars import MODEL_DIR, MAX_SHARD_SIZE


def main():
    log_level = -1
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=False, choices=["7b", "13b", "70b", "7b-chat", "13b-chat", "70b-chat"]
    )
    args = parser.parse_args()

    model_name = "Llama-2-70b-hf" if args.model is None else f"Llama-2-{args.model}-hf"
    model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{model_name}")
    model.save_pretrained(
        MODEL_DIR / "model" / "pretrained" / model_name, max_shard_size=MAX_SHARD_SIZE
    )


if __name__ == "__main__":
    main()
