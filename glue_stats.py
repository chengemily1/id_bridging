from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM

import numpy as np

import torch
import torch.nn.functional as F


from tqdm import tqdm
import os
import json
import argparse
from argparse import Namespace
import types

from dataset_util import process_dataset, HF_TOKEN
from arguments import *
from accelerate import Accelerator

"""
Get some stats about GLUE datasets:

** Low levels
* size: the number of data samples
* vocab_size: the size of the vocabulary
* vocab_ent: the entropy of the words distribution to measure the balanceness of the dataset
* avg_input_length: the average number of words per data samples

** High levels
* ?

"""
IGNORED_TOKENS=[2,] # EOS token for OPT

parser = argparse.ArgumentParser(description='OPTCorpus generation')

# Data selection
parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--sample_size', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--path_optcorpus')
parser.add_argument('--save_per_sample', action='store_true', help='Save per sample stats')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--fp16', action='store_true', help='use half precision')
args = parser.parse_args()
print(args)

def flatten(l): # recursive
    flattened = [item for sublist in l for item in sublist]
    return flattened if not isinstance(flattened[0], list) else flatten(flattened)

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def get_size(d):
    return sum([d_split.num_rows for split, d_split in data.items()])

def add_to_vocab(vocab, word):
    if word not in IGNORED_TOKENS:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1

def entropy(v, norm=False):
    if norm:
        v = v / v.sum()
    h = -np.sum(v * np.log(v))
    return h

def nll(logits, label_ids):
    label_ids = label_ids.unsqueeze(-1)
    predict_logp = F.log_softmax(logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    return -target_logp.squeeze()

def shift_batch_tensor(t):
    return torch.concat([t[:,1:], t[:,0].unsqueeze(-1)], dim=1)

def get_perplexity(encodings, model):
    input_ids = encodings.input_ids.to(model.device)
    attention_mask = encodings.attention_mask.to(model.device)
    target_ids = input_ids.clone()
    target_ids = torch.where(target_ids==1, 0, target_ids) # ignore padding tokens
    # shift targets, as each position predicts the next token
    target_ids[:,0] = 0 # ignore the first token as it is an eos
    target_ids = shift_batch_tensor(target_ids)
    mask = 1 * target_ids.eq(0) # 1 means the position has to be masked

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits
        loss = nll(logits, target_ids)
        # mask padding tokens + average over valid tokens
        loss = (loss * (1-mask)).sum(-1) / (1-mask).sum(-1)

    ppl = torch.exp(loss.float()).to('cpu')
    return ppl

stats = {}

accelerator = Accelerator(cpu=args.cpu)

# Get model, tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
if 'Llama' in args.model_name: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    load_in_8bit=True,
    torch_dtype=torch.float16 if args.fp16 else torch.float32,
    token=HF_TOKEN
    device_map="auto"
)
model.config.n_positions = model.config.max_position_embeddings
model = accelerator.prepare(model)

# List corpora to process (benchmark, task_name)
corpuses = []
corpuses += [('glue',subset) for subset in task_to_keys['glue']]
corpuses += [('super_glue',subset) for subset in task_to_keys['super_glue']]
corpuses += [('', 'wikitext'), ]
corpuses += [('', 'tweet_eval'), ('', 'NeelNanda/pile-10k'), ('', 'cnn_dailymail'),]
corpuses += [('', 'stas/openwebtext-10k'),]

if args.save_per_sample:
    per_sample_stats = {}

for corpus, subset in corpuses:

    print(f'[{corpus}.{subset}] Starting...')

    vocab_count = {}
    parag_lengths_list = []
    parag_ppl_list = []

    # 1) Load data as a list of paragraphes
    arguments = Namespace(
        model_name=args.model_name,
        benchmark_name=corpus,
        task_name=subset,
        sample_size=args.sample_size,)
    N, paragraphs = process_dataset(arguments, remove_empty=True)

    print("Dataset size:", len(paragraphs))

    # 2) Batch paragraphs
    batches = [paragraphs[i:i+args.batch_size] for i in range(0,len(paragraphs),args.batch_size)]

    # 3) Iterate and compute stats on each batch
    cpt_out = 0
    for batch in tqdm(batches, desc=f'[{subset}] Iterating'):
        if type(batch[0]) is str:
            sentences_tkn = tokenizer(batch, truncation=True, max_length=model.config.n_positions, return_tensors="pt", padding=True)
        else: # already tokenized, must add padding tokens
            sentences_tkn = types.SimpleNamespace()
            # nested is not compatible with old pytorch versions
            max_length = max([len(t) for t in batch])
            setattr(sentences_tkn, 'input_ids', torch.stack([F.pad(torch.tensor(t), (0,max_length-len(t)), value=tokenizer.pad_token_id) for t in batch]))
            setattr(sentences_tkn,'attention_mask', torch.where(sentences_tkn.input_ids.eq(1),0,1))
        parag_lengths_list += sentences_tkn.attention_mask.sum(-1).tolist()
        [add_to_vocab(vocab_count, tkn) for tkn in torch.masked_select(sentences_tkn.input_ids, sentences_tkn.attention_mask.bool()).tolist()]
        ppl = get_perplexity(sentences_tkn, model)
        if ppl.isnan().any():
            print('Nan detected!')
            print(sentences_tkn)
            print(batch)
        parag_ppl_list += ppl.tolist()

    print("Number of samples:", len(parag_ppl_list))

    if args.save_per_sample:
        model_str = args.model_name.split('/')[-1]
        dataset_str = '_'.join([corpus.split('/')[-1],subset.split('/')[-1]])
        with open(os.path.join('stats', f'{model_str}_{dataset_str}_per_sample_ppl.csv'), 'w') as f:
            f.write(','.join(['perplexity', 'length (tokens)'])+'\n')
            for (p,l) in zip(parag_ppl_list, parag_lengths_list):
                f.write(','.join([str(p), str(l)])+'\n')
    else:
        stats[subset] = {
            'size'          : len(paragraphs),
            'vocab_size'    : len(vocab_count),
            'vocab_ent'     : entropy(np.array(list(vocab_count.values())), norm=True),
            'parag_length'  : {'avg': np.mean(parag_lengths_list), 'var': np.var(parag_lengths_list)},
            'parag_ppl'     : {'avg': np.mean(parag_ppl_list), 'var': np.var(parag_ppl_list)},
        }

        print(f'[{subset}] Done.')
        print(stats)
if not args.save_per_sample:
    print(stats)
