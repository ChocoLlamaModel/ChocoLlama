# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import logging
import os
import math
import pathlib
from pkg_resources import packaging

import fire
import torch
from torch.distributed.elastic.multiprocessing.errors import record as record_torch
from torch.distributed.elastic.multiprocessing.errors import ErrorHandler
import torch.distributed as dist
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizerFast,
    RobertaTokenizerFast,
    LlamaConfig,
    MixtralForCausalLM,
    MixtralConfig,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

from llama_recipes.configs import fsdp_config
from llama_recipes_patch.configs import train_config
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes_patch.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from llama_recipes_patch.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes_patch.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)
from llama_recipes_patch.model_checkpointing import load_model_sharded, load_optimizer_checkpoint

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def get_layer_num(layer):
    layer = layer.split('.')
    if 'layers' in layer:
        return int(layer[layer.index('layers')+1])
    else:
        return -1

def main(**kwargs):
    os.environ["NCCL_DEBUG"] = "INFO"
    logging.info('Setting up training run.')
    logging.info(kwargs)
    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    checkpoint_dir = None
    if train_config.resume_training:
        if os.path.exists(train_config.output_dir):
            checkpoint_dirs = os.listdir(train_config.output_dir)
            if len(checkpoint_dirs) > 0:
                most_recent_ckpt = max(checkpoint_dirs)
                checkpoint_dir = os.path.join(train_config.output_dir,most_recent_ckpt)
                logging.info(f'Loading weights from checkpoint: {checkpoint_dir}')

    if "Llama" in train_config.model_name:
        causal_lm_class = LlamaForCausalLM
        model_config_class = LlamaConfig
        if "translated" in train_config.model_name:
            tokenizer_class = RobertaTokenizerFast
        else:
            tokenizer_class = LlamaTokenizerFast
        decoder_layer_class = LlamaDecoderLayer
    elif "Mixtral" in train_config.model_name:
        causal_lm_class = MixtralForCausalLM
        model_config_class = MixtralConfig
        tokenizer_class = LlamaTokenizerFast
        decoder_layer_class = MixtralDecoderLayer
    else:
        raise ValueError(f'Unsupported model: {train_config.model_name}')

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = causal_lm_class.from_pretrained(
                train_config.model_name,
                torch_dtype=torch.bfloat16 if fsdp_config.pure_bf16 else None,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
        else:
            model_config = model_config_class.from_pretrained(train_config.model_name)
            model_config.use_cache = use_cache
            with torch.device("meta"):
                model = causal_lm_class(model_config)

    else:
        model = causal_lm_class.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
        )
    if train_config.enable_fsdp and train_config.use_fast_kernels and (not "Mixtral" in train_config.model_name):
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model) 
        except ImportError:
            logging.warning("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # Load the tokenizer and add special tokens
    tokenizer = tokenizer_class.from_pretrained(train_config.tokenizer_name)
    tokenizer.add_special_tokens(
            {

                "pad_token": "<PAD>",
            }
        )
    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        logging.info(f'Training with: {train_config.peft_method}')
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # all_params = sum(p.numel() for p in model.parameters())
        # logging.info(f'trainable params: {trainable_params} || all params: {all_params} || trainable%: {trainable_params/all_params*100}')
        model.print_trainable_parameters()
    else:
        num_layers = max([get_layer_num(layer) for layer in list(model.state_dict().keys())])+1
        num_trainable_layers = 2*math.floor(num_layers*train_config.fraction_layers_trainable/2)
        untrainable_layers_start = num_trainable_layers//2
        untrainable_layers_end = num_layers-num_trainable_layers//2-1
        for param_name, param in model.named_parameters():
            layer_num = get_layer_num(param_name)
            if ((layer_num >= 0 and layer_num < untrainable_layers_start)
                or (layer_num > untrainable_layers_end)
                or (train_config.embeddings_trainable and '.embed_tokens.' in param_name)
                or (train_config.train_head and 'lm_head' in param_name)
                ):
                param.requires_grad = True
                logging.info(f'🧠 {param_name}')
            else:
                param.requires_grad = False
                logging.info(f'🧊 {param_name}')
        
        trainable_params = sum([layer.numel() for layer in model.parameters() if layer.requires_grad])
        total_params = sum([layer.numel() for layer in model.parameters()])
        logging.info('Training with continued pretraining')
        logging.info(f'{trainable_params/1e9:.3f}B out of {total_params/1e9:.3f}B parameters trainable ({trainable_params/total_params*100:.2f}%)')

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, decoder_layer_class)

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=not train_config.use_peft,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif train_config.cpu_training:
        model.to("cpu")
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training
    dataset_train = get_preprocessed_dataset(
        train_config.tokenizer_name,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        logging.info(f"--> Training Set Length = {len(dataset_train)}")

    train_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=False,
        )

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    learning_rate = train_config.lr
    if checkpoint_dir is not None:
        state_dict_path = os.path.join(checkpoint_dir,'other_state.pt')
        checkpoint = torch.load(state_dict_path)
        learning_rate = checkpoint['lr_scheduler_state_dict']['_last_lr'][0]

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=learning_rate,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
        )

    scheduler = StepLR(optimizer, step_size=train_config.lr_step_size, gamma=train_config.gamma)

    first_epoch = 0
    first_step = 0

    if checkpoint_dir is not None:
        logging.info(f'Loading model and optimizer state: {checkpoint_dir}')
        load_model_sharded(checkpoint_dir,model,rank)
        load_optimizer_checkpoint(model,pathlib.Path(checkpoint_dir)/'optimizer.pt',rank)

        logging.info(f'Loading scheduler and epoch state: {state_dict_path}')
        scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        first_epoch = checkpoint['next_epoch']
        first_step = checkpoint['next_step']

    # Start the training process
    results = train(
        model,
        train_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        first_epoch,
        first_step,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [logging.info(f'Key: {k}, Value: {v}') for k, v in results.items()]

main = record_torch(fn=main,error_handler=ErrorHandler())
if __name__ == "__main__":
    fire.Fire(main)
