import datasets
import tqdm
import transformers
from transformers import LlamaTokenizerFast
from datasets import Dataset

import argparse
import sys
import os

import logging
from global_vars import DATA_DIR, MODEL_DIR
from datasets import concatenate_datasets

def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = -1
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True) # eg "bigscience/bloom-560m"
    parser.add_argument(
        "--strategy", type=str, choices=["original", "new", "extended"], required=True
    )
    parser.add_argument("--new_vocab_size", required=False, type=int)
    parser.add_argument("--datasets", required=False, type=str)

    # load arguments
    args = parser.parse_args()

    # load pretrained tokenizer
    model_name = args.model.split("/")[-1]
    if model_name.startswith('Mixtral'):
        tokenizer_class = LlamaTokenizerFast
    elif model_name.startswith('Llama-2'):
        tokenizer_class = LlamaTokenizerFast
    else:
        raise ValueError('Only Llama-2 tokenizers supported.')
    tokenizer = tokenizer_class.from_pretrained(args.model, use_auth_token=True)

    # if just saving the original one, don't do anything else
    if args.strategy == "original":
        new_tokenizer = tokenizer
        tokenizer_name = f"{model_name}_{args.strategy}"

    else:
        # if adapting the tokenizer, load data
        input_dir = DATA_DIR / "processed_raw"

        if args.datasets is not None:
            dataset_names = args.datasets.split(',')
            dataset_name_for_saving = args.datasets
        else:
            dataset_names = os.listdir(input_dir)
            dataset_name_for_saving = 'all'

        raw_datasets = []
        for dataset_name in dataset_names:
            dataset = Dataset.load_from_disk(input_dir / dataset_name / "train")
            raw_datasets.append(dataset)

        raw_datasets = concatenate_datasets(raw_datasets)

        logger.info(f"Training tokenizer on these datasets: {dataset_names}")

        def batch_iterator():
            batch_size = 1000
            for i in tqdm.tqdm(range(0, len(raw_datasets), batch_size)):
                sample = raw_datasets[i: i + batch_size]["text"]
                yield sample

        original_tokenizer_vocab_size = len(tokenizer)
        new_vocab_size = args.new_vocab_size or original_tokenizer_vocab_size
        tokenizer_name = f"{model_name}_{args.strategy}_{dataset_name_for_saving}_{new_vocab_size}"

        # extend tokenizer by training a new one and adding new tokens to the original one
        if args.strategy == "extended":
            new_tokenizer = tokenizer.train_new_from_iterator(
                batch_iterator(), vocab_size=new_vocab_size
            )
            added = tokenizer.add_tokens([tok for tok in new_tokenizer.vocab.keys()])
            logger.info(f"Overlap with previous vocab: {len(new_tokenizer) - added}")
            logger.info(f"Added to new vocab: {added}")
            new_tokenizer = tokenizer

        # create an entirely new tokenizer
        elif args.strategy == "new":
            logger.info(f"Training new tokenizer {tokenizer_name}")
            new_tokenizer = tokenizer.train_new_from_iterator(
                batch_iterator(), vocab_size=new_vocab_size
            )

    logger.info(f"✅ Got tokenizer with len {len(new_tokenizer)}")

    # for fun: print some tokens
    vocab = new_tokenizer.get_vocab()
    print('20 tokens from the vocabulary of the tokenizer: ', list(vocab.keys())[:20])

    new_tokenizer.save_pretrained(MODEL_DIR / "tokenizer" / tokenizer_name)

if __name__ == "__main__":
    main()
