# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="PATH/to/LLAMA/7B"
    tokenizer_name: str="PATH/to/LLAMA/7B"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    save_freq_steps: int=100
    batch_size_training: int=4
    gradient_accumulation_steps: int=1
    num_epochs: int=3
    num_workers_dataloader: int=1
    lr: float=1e-4
    lr_step_size: int=1000
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    dataset = "dutch_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    latest_model_name: str = "latest"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    cpu_training: bool = False
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    checkpointing_safety_margin: int = 30*60 # Number of seconds before the end of a SLURM job when we should wind down and start checkpointing. 
    slurm_job_path: str | None = None # Path to the slurm job to be triggered for continuation
    resume_training: bool = False
    embeddings_trainable: bool = False # Whether to train embeddings during continued pretraining
    train_head: bool = False # Whether to train the head during continued pretraining.
    fraction_layers_trainable: float = 1 # Fraction of hidden layers to train during continued pretraining (equally distributed at the back and front of the model)
    
    
    
