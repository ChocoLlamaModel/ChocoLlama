#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
# sbatch ./scripts/long_run/7b-LoRa-original.slurm

#SBATCH --job-name=7b-lora-original-2

#SBATCH --mem=200G                   # amount of memory requested per node
#SBATCH --time 72:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=./.output/%x-%j.out # output file name

#PBS -l nodes=2:gpus=4
#PBS -A 2023_079

export WANDB_EXPERIMENT_NAME=$SLURM_JOB_NAME
echo Running experiment $WANDB_EXPERIMENT_NAME
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# Enable for A100
export FI_PROVIDER="efa"

echo Node IP: $head_node_ip
echo Nodes: $SLURM_JOB_NODELIST
export LOGLEVEL=INFO
# debugging flags (optional)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INFO
export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# llama-recipes provided that we might need these on our cluster, but if we include them it does not work
# so just comment them out and it does
# export NCCL_SOCKET_IFNAME="ens"
# export FI_EFA_USE_DEVICE_RDMA=1

eval "$(conda shell.bash hook)"
conda activate chocollama

# Total samples: 7950719
# -> 496 919 steps @ batch size 16
# -> 62 115 steps @ 8 GPUs
srun torchrun --nnodes $SLURM_JOB_NUM_NODES --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 finetuning.py  \
    --model_name {PATH_TO_MODELS}/Llama-2-7b-hf \
    --tokenizer_name  {PATH_TO_TOKENIZERS}/Llama-2-7b-hf \
    --batch_size_training 16 \
    --save_freq_steps 3106 \
    --gradient_accumulation_steps 8 \
    --num_epochs 1 \
    --num_workers_dataloader 4 \
    --lr 0.0003 \
    --lr_step_size 6212 \
    --dataset all_training_data \
    --output_dir {PATH_TO_MODELS}/adapted/Llama-2-7b-hf/lora-original \
    --modules_to_save "[embed_tokens,lm_head]" \
    --use_peft --peft_method lora \
    --use_fast_kernels \
    --pure_bf16 \
    --low_cpu_fsdp \
    --slurm_job_path {PATH_TO_CHOCOLLAMA}/scripts/long_run/7b-LoRa-original.slurm \
    --resume_training \
    --enable_fsdp
