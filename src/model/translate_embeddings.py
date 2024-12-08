import json
import torch
import transformers
import argparse
from functools import reduce
from typing import Tuple
from tqdm import tqdm
from global_vars import MODEL_DIR, MAX_SHARD_SIZE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--og_tokenizer", type=str, required=True)
    parser.add_argument("--og_model", type=str, required=True)
    parser.add_argument("--new_model_name", type=str, required=True)
    parser.add_argument("--path_to_mapping", type=str, required=True)
    parser.add_argument("--dutch_tokenizer_name", type=str, default='FremyCompany/roberta-base-nl-oscar23')
    args = parser.parse_args()

    OLD_LANGUAGE='en'
    NEW_LANGUAGE='nl'
    
    print("Loading tokenizer")
    # load tokenizers for the two models
    old_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer" / args.og_tokenizer)
    new_tokenizer = transformers.AutoTokenizer.from_pretrained(args.dutch_tokenizer_name)

    # load the old model
    print("Loading old model")
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_DIR / "model" / "pretrained" / args.og_model)

    # get the mapping between the two tokenizers
    mapping : list[Tuple[str, list[Tuple[str, float]]]] = []
    with open(args.path_to_mapping, 'r') as f:
        mapping : list[Tuple[str, list[Tuple[str, float]]]] = json.load(f)

    # disable gradient computation temporarily
    print("Translating embeddings")
    with torch.no_grad():

        # get the embeddings of the OLM model
        old_embeddings = model.get_input_embeddings()
        old_output_embeddings = model.get_output_embeddings()

        # change the tokenizer of the OLM model to the one of the RobBERT model, and reinitialize the embeddings
        model.resize_token_embeddings(1)  # this is a hack to make the model forget its old tokenizer
        model.resize_token_embeddings(len(new_tokenizer))  # this is the actual call to change the tokenizer
        new_embeddings = model.get_input_embeddings()
        new_output_embeddings = model.get_output_embeddings()
        model.config.vocab_size = len(new_tokenizer)
        model.config.pad_token_id = new_tokenizer.pad_token_id
        model.config.bos_token_id = new_tokenizer.bos_token_id
        model.config.eos_token_id = new_tokenizer.eos_token_id
        model.config.unk_token_id = new_tokenizer.unk_token_id
        model.config.sep_token_id = new_tokenizer.sep_token_id
        model.config.cls_token_id = new_tokenizer.cls_token_id
        model.config.mask_token_id = new_tokenizer.mask_token_id
        model.config.additional_special_tokens_ids = new_tokenizer.additional_special_tokens_ids
        model.config.tokenizer_class = new_tokenizer.__class__.__name__

        # for each token in the new tokenizer, find the corresponding tokens in the old tokenizer, and average their embeddings
        for new_id in tqdm(range(len(new_tokenizer))):
            old_tokens = mapping[new_id][1] # list of (ids,weight) in the old tokenizer

            # sort the list such that the smallest weights come first (for numerical stability)
            old_tokens = sorted(old_tokens, key=lambda x: x[1])

            # map tokens to their ids
            old_ids = [(old_tokenizer.convert_tokens_to_ids(old_token), weight) for old_token, weight in old_tokens]

            # we use a weighted average, where the first token in the list has 0.4 weight, the second 0.2, and the remaining 0.4 are equally distributed among all tokens (including the first two)
            if len(old_ids) > 1:
                new_embeddings.weight.data[new_id] = reduce(lambda a, b: a.add_(b), [old_embeddings.weight.data[old_id]*weight for old_id, weight in old_ids])
                new_output_embeddings.weight.data[new_id] = reduce(lambda a, b: a.add_(b), [old_output_embeddings.weight.data[old_id]*weight for old_id, weight in old_ids])
            elif len(old_ids) == 1:
                new_embeddings.weight.data[new_id] = old_embeddings.weight.data[old_ids[0][0]]
                new_output_embeddings.weight.data[new_id] = old_output_embeddings.weight.data[old_ids[0][0]]
            else: # use the unknown token embedding if the token is not in the old tokenizer
                new_embeddings.weight.data[new_id] = old_embeddings.weight.data[old_tokenizer.unk_token_id]
                new_output_embeddings.weight.data[new_id] = old_output_embeddings.weight.data[old_tokenizer.unk_token_id]

    # save the model
    print("Saving results")
    model.save_pretrained(MODEL_DIR / "model" / "pretrained" / args.new_model_name, max_shard_size=MAX_SHARD_SIZE)
    new_tokenizer.save_pretrained(MODEL_DIR / "tokenizer" / f"{args.new_model_name}_mapping_{OLD_LANGUAGE}-{NEW_LANGUAGE}")

if __name__ == "__main__":
    main()
