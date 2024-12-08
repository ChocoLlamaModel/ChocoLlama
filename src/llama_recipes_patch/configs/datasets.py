# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

     
@dataclass
class dutch_dataset:
    dataset: str =  "dutch_dataset"
    train_split: str = "train"
    test_split: str = "test"
    input_length: int = 4096
    max_samples: int | None = None

@dataclass
class dataset_10perc:
    dataset: str =  "dataset_10perc"
    train_split: str = "train"
    test_split: str = "test"
    input_length: int = 4096
    max_samples: int | None = None

@dataclass
class all_training_data:
    dataset: str =  "all_training_data"
    train_split: str = "train"
    test_split: str = "test"
    input_length: int = 4096
    max_samples: int | None = None
