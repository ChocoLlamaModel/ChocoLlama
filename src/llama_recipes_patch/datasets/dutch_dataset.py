# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets
from global_vars import DATA_DIR

from llama_recipes_patch.datasets.utils import Concatenator

def get_preprocessed_dutch(dataset_config, tokenizer, split):
    tokenizer = str(tokenizer).split('/')[-1]
    input_dir = DATA_DIR / "LM_tokenized" / tokenizer / "Wikipedia_20230301_NL"
    dataset = datasets.load_from_disk(input_dir / split)
    dataset = dataset.shuffle(seed=42)
    if dataset_config.max_samples is not None:
        return dataset.select(range(len(dataset))[:dataset_config.max_samples])
    return dataset

def get_preprocessed_10perc(dataset_config, tokenizer, split):
    tokenizer = str(tokenizer).split('/')[-1]
    input_dir = DATA_DIR / "LM_tokenized" / tokenizer / "all_data_10perc"
    dataset = datasets.load_from_disk(input_dir)
    dataset = dataset.shuffle(seed=42)
    if dataset_config.max_samples is not None:
        return dataset.select(range(len(dataset))[:dataset_config.max_samples])
    return dataset

def get_all_training_data(dataset_config, tokenizer, split):
    tokenizer = str(tokenizer).split('/')[-1]
    input_dir = DATA_DIR / "LM_tokenized" / tokenizer / "all_training_data"
    dataset = datasets.load_from_disk(input_dir)
    #dataset = dataset.shuffle(seed=42) # is already shuffled
    if dataset_config.max_samples is not None:
        return dataset.select(range(len(dataset))[:dataset_config.max_samples])
    return dataset

