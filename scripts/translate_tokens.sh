#/usr/bin/env bash

#PBS -l nodes=1:ppn=20
#PBS -l walltime=10:00:00
#PBS -l mem=480gb
#PBS -A starting_2023_020

eval "$(conda shell.bash hook)"
conda activate llama_2_nl

python src/model/translate_embeddings.py --og_tokenizer="Llama-2-7b-hf_original" --og_model="Llama-2-7b-hf" --new_model_name="Llama-2-7b-hf-translated-embeddings" --path_to_mapping="/data/gent/458/vsc45861/GitFolder/Llama-2-NL/src/model/token_translation_mapping/en-nl.json"
