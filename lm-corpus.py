import os
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import numpy as np
import argparse
import transformers
from transformers import (
    AutoTokenizer,
    OPTForCausalLM,
    AutoModelForCausalLM,
    pipeline,
    set_seed,
)
import json
import math
from tqdm import tqdm
from arguments import *
import json
import random
from accelerate import Accelerator
import types
import gc


FIELD_NAMES = ['generated_text', 'generated_token_ids']
DEFAULT_MAX_WINDOW = 2048

def init_model(args):

    # accelerate
    accelerator = Accelerator(cpu=args.cpu)

    model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
            # device_map="auto"
    )

    model = accelerator.prepare(model)

    max_window_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else DEFAULT_MAX_WINDOW

    print(f"Max window length: {max_window_length}")

    return model, accelerator, max_window_length

def generate_tokens(model, tokenizer, tkns_input, n_tokens):
    max_length = max([len(t) for t in tkns_input])
    assert all([tokenizer.eos_token_id not in t[1:] for t in tkns_input])
    input_ids = torch.stack([F.pad(t, (max_length-len(t), 0), value=tokenizer.pad_token_id) for t in tkns_input])
    attention_mask = torch.where(input_ids.eq(tokenizer.pad_token_id),0,1)
    l = [max_length - len(t) for t in tkns_input]
    ar = torch.arange(len(attention_mask))
    attention_mask[ar,l] += 1 # small fix to not mask the bos token
    # generate
    tokens_generated = model.generate(
        input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device), max_new_tokens=n_tokens, do_sample=True, top_k=0,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    # unpad
    tokens_generated = [tokens_generated[i,l[i]:] for i in range(len(tokens_generated))]
    return tokens_generated

def pick_batch_and_generate(model, tokenizer, prefix_stack, batch_size, n_tokens, pbar, max_window_length):
    # takes the batch_size first strings of the prefix_stack
    generate_input = []
    for _ in range(batch_size):
        try:
            generate_input += [prefix_stack.pop(0),]
        except IndexError:
            # batch can't be full, not enough prefix
            break

    tkns_input = [inpt['tokens'].clone() for inpt in generate_input] # make a copy
    batch_max_input_length = max([inpt['len'] for inpt in generate_input])
    # length
    pbar.set_description(f"[Generate until EOS][BZ: {batch_size}][Max input length: {batch_max_input_length}] Completed sentences:")


    # make sure that we do not exceed the LM window
    troncated_tkns = None
    if batch_max_input_length+args.batch_max_tkn_length+1 > max_window_length:
        diff = batch_max_input_length+args.batch_max_tkn_length+1-max_window_length
        tkns_input = [t[diff:] for t in tkns_input]
        troncated_tkns = [t[:diff] for t in tkns_input]

    try:
        with torch.no_grad():
            tokens_generated = generate_tokens(model, tokenizer, tkns_input, n_tokens)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            # put back current input into the prefix stack
            prefix_stack += generate_input
            raise e
        else:
            raise e

    # add troncated part
    if troncated_tkns is not None:
        tokens_generated = [torch.cat([troncated_tkns[i], t], dim=-1) for i,t in enumerate(tokens_generated)]

    return tokens_generated

def remove_eos_sentences(prefix_stack, finished, tokenizer, tokens_generated, eos_id, max_parag_length):
    # remove generated text having reached eos
    if max_parag_length > 0:
        text_with_eos = [(len(tkns) > 1 and tkns[-1] == eos_id) or (len(tkns)>=max_parag_length) for tkns in tokens_generated]
    else:
        text_with_eos = [len(tkns) > 1 and tkns[-1] == eos_id for tkns in tokens_generated]

    finished += [{
        'generated_text': tokenizer.decode(tokens_generated[i], skip_special_tokens=True),
        'generated_token_ids': tokens_generated[i].cpu().tolist(),
        } for i,is_eos in enumerate(text_with_eos) if is_eos]
    # put the other ones in the prefix_stack
    prefix_stack += [{'tokens':tokens_generated[i], 'len': len(tokens_generated[i])}\
                        for i,is_eos in enumerate(text_with_eos) if not is_eos]
    n_new_finished = sum(text_with_eos)
    if n_new_finished > 0:
        print('[COMPLETED SAMPLE] ',finished[-1]['generated_text'])
    return prefix_stack, finished, n_new_finished

def generate_until_eos(n_samples, batch_size=8, max_parag_length=-1, max_window_length=2048):

    use_fast_tokenizer = 'opt' not in args.model_name # because of OPT tokenizer
    if 'llama' in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            use_fast=False,
            padding_side="left",
            use_auth_token=HF_TOKEN_LLAMA2
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=use_fast_tokenizer, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    eos = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    bos = tokenizer.bos_token
    bos_id = tokenizer.bos_token_id

    # progression bar
    pbar = tqdm(total=n_samples)
    pbar.set_description(f"[Generate until EOS][BZ: {batch_size}][Max input length: {1}] Completed sentences:")


    # init_prefix = '' if 'opt' in args.model_name else bos
    prefix_stack = [{'tokens':torch.tensor([bos_id]), 'len': 1}] *n_samples # len is 1 because of the eos
    finished = []

    # init model, move it to the device
    model, accelerator, max_window_length = init_model(args)
    # move prefix to the device
    prefix_stack = [{'tokens':p['tokens'].to(model.device), 'len': p['len']} for p in prefix_stack]

    while len(prefix_stack) > 0:

        # handle OOM by reducing batch size
        try:
            tokens_generated = pick_batch_and_generate(model, tokenizer, prefix_stack, batch_size, args.batch_max_tkn_length, pbar, max_window_length)
            prefix_stack, finished, n_new_finished = remove_eos_sentences(prefix_stack, finished, tokenizer, tokens_generated, eos_id, max_parag_length)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f'[OOM] Freeing memory')
                # free model
                model = model.to('cpu')
                del model
                del accelerator
                # move data to cpu
                prefix_stack = [{'tokens':p['tokens'].to('cpu'), 'len': p['len']} for p in prefix_stack]
                # free memory
                gc.collect()
                torch.cuda.empty_cache()
                # reinit model and put data into the devoce
                model, accelerator, max_window_length = init_model(args)
                prefix_stack = [{'tokens':p['tokens'].to(model.device), 'len': p['len']} for p in prefix_stack]
                # decrease batch size
                new_batch_size = int(batch_size * 0.8)
                if new_batch_size == 0:
                    print(f'[OOM] even batch-size=1 does not fit')
                    print(f'[OOM] Stop here.')
                    return finished
                print(f'[OOM] Reducing batchsize from {batch_size} to {new_batch_size}')
                batch_size = new_batch_size
            else:
                raise e

        # update progression bar
        pbar.update(n_new_finished.item())

    pbar.close()
    return finished


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='OPTCorpus generation')

    # Data selection
    parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--max_parag_length', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_max_tkn_length', type=int, default=8)
    parser.add_argument('--path_to_results_cache', default='/home/echeng/adapters_id/optcorpus/')
    parser.add_argument('--ignore_cache', type=bool, default=False)
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--fp16', action='store_true', help='use half precision')

    args = parser.parse_args()
    print(args)

    # set random seed
    seed = random.randint(0, 999999)
    print(f'Seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # hugging face
    set_seed(seed)

    # Create results file folder if doesn't already exist
    args.path_to_results_cache += args.model_name
    if not os.path.exists(args.path_to_results_cache):
        os.makedirs(args.path_to_results_cache)

    results = []

    # Each result is a "document"
    results = generate_until_eos(n_samples=args.n_samples, batch_size=args.batch_size, max_parag_length=args.max_parag_length)

    # Open json file in append mode
    # And write the results
    with open(args.path_to_results_cache + f'/seed_{seed}.jsonl', 'a') as f:
        for d in results:
            json.dump(d,f)
            f.write('\n')
