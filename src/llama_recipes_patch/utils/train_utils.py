# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import logging
import dataclasses
import datetime
import subprocess
import os
import time
import yaml
from pathlib import Path
from pkg_resources import packaging


import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import wandb
from global_vars import WANDB_LOGGING_DIR

from llama_recipes_patch.model_checkpointing import save_model_and_optimizer_sharded, save_optimizer_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen_mixed, get_llama_wrapper
from llama_recipes_patch.utils.memory_utils import MemoryTrace
from general_utils.get_seconds_left import get_seconds_left

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.info('Logging started')

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)

def train(model, train_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps,
          train_config, first_epoch, first_step, fsdp_config=None, local_rank=None, rank=None):
    """
    Trains the model on the given dataloader
    
    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        first_epoch: The epoch to resume from.
        first_step: The step to resume from.
        tokenizer: tokenizer used in the eval for decoding the predicitons
    
    Returns: results dictionary containing average training and validation perplexity and loss
    """

    # set up wandb
    while True and rank == 0:
        try:
            wandb.init(
                # set the wandb project where this run will be logged
                project="llama-2_NL",
                id=os.environ["WANDB_EXPERIMENT_NAME"],
                group=os.environ["WANDB_EXPERIMENT_NAME"],
                dir=WANDB_LOGGING_DIR,
                resume=True,
                # track hyperparameters and run metadata
                config=dataclasses.asdict(train_config())
            )
            break
        except Exception as e:
            logging.error(e)
            raise e
            time.sleep(60)


    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler() 
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"]) 
    train_prep = []
    train_loss = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    for epoch in range(first_epoch,train_config.num_epochs):
        logging.info(f'Starting epoch {epoch} at step {first_step}.')
        epoch_start_time = time.perf_counter()
        with MemoryTrace(gpu=not train_config.cpu_training) as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            if rank == 0:
                pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length)
            for step, batch in enumerate(train_dataloader):
                if step < first_step:
                    if rank == 0:
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            pbar.update(1)
                    continue
                for key in batch.keys():
                    if train_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    elif train_config.cpu_training:
                        batch[key] = batch[key].to('cpu')
                    else:
                        batch[key] = batch[key].to('cuda:0')              
                loss = model(**batch).loss
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if rank == 0:
                            pbar.update(1)
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        if rank == 0:
                            pbar.update(1)
                
                if rank == 0:
                    wandb.log({"train_epoch": epoch/train_config.num_epochs,
                           "train_step": step/len(train_dataloader),
                           "train_loss": loss.detach().float() * gradient_accumulation_steps,
                           "learning_rate": lr_scheduler.get_last_lr()[0]})
                    pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, "
                                         f"step {step}/{len(train_dataloader)} completed "
                                         f"(loss: {loss.detach().float() * gradient_accumulation_steps})")

                # Update the learning rate as needed
                lr_scheduler.step()

                # let's save whenever:
                #  - there have been save_freq_steps minibatches processed (batch size x world size)
                #  - we've reached the end of the epoch
                #  - we're nearing the end of the slurm job lifetime
                save_freq_steps_reached = ((step + 1) % int(train_config.save_freq_steps) == 0)
                end_of_epoch_reached = step >= len(train_dataloader)-1
                end_of_job_reached = ((step + 1) % gradient_accumulation_steps == 0) and (get_seconds_left() <= train_config.checkpointing_safety_margin)
                if (save_freq_steps_reached
                    or end_of_job_reached
                    or end_of_epoch_reached):
                    logging.info(f'Checkpointing after {step} steps.')
                    
                    checkpoint_start_time = time.perf_counter()
                    
                    if train_config.enable_fsdp:
                        dist.barrier()
                    logging.info('Saving latest model checkpoint.')
                    checkpoint_path = f'{train_config.output_dir}/{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}'
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path,exist_ok=True)
                    if rank == 0:
                        torch.save({
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            'next_epoch':epoch if not end_of_epoch_reached else epoch+1,
                            'next_step':step+1 if not end_of_epoch_reached else 0
                            },f'{checkpoint_path}/other_state.pt')
                    save_model_and_optimizer_sharded(checkpoint_path, model, rank, optim=optimizer)
                    save_optimizer_checkpoint(
                            checkpoint_path, model, optimizer, rank, epoch=epoch
                        )

                    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                    checkpoint_times.append(checkpoint_end_time)
                    if train_config.enable_fsdp:
                        dist.barrier()


                # Exit when nearing the end of the job to avoid spending compute
                # on unsaved weight updates.
                if end_of_job_reached:
                    # Launch a new job to resume whenever the end of the job has been reached.
                    # logging.info('Nearing the end of the job. Continuing from new job...')
                    # if not end_of_epoch_reached and train_config.slurm_job_path is not None and rank == 0:
                    #     command = [
                    #         'sbatch',
                    #         train_config.slurm_job_path
                    #     ]
                    #     subprocess.run(command)
                    logging.info('Nearing the end of the job. Continuing from new job...')
                    exit(1)

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)    
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)
        
        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        else:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")

        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s")

        first_step = 0
    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    
    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)
        
    return results

def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")
                
                
def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True' 
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")




def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""
    
    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries, 
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")
