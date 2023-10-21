import torch
import os
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import skdim

import json
import argparse
import pandas as pd
import random
import sys
from arguments import *
from dataset_util import process_dataset, encode_data

def plot_ids(args, ids):
    plt.plot(np.arange(len(ids)) / len(ids), ids, 'o-')
    plt.ylabel('Intrinsic Dimension', fontsize=18)
    plt.xlabel('Relative Layer Depth ({})'.format(args.model_name.split('/')[-1]), fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('{} Representational ID'.format(args.task_name.upper()), fontsize=18)
    plt.tight_layout()
    plt.savefig('{}_id_{}_{}.pdf'.format(args.task_name, args.model_name.split('/')[-1], args.pooling))

####### REPRESENTATION AGGREGATION METHODS ###############
def resize_tensor(x):
    batch_size = x.shape[0]
    extrinsic_dim = x.shape[1] * x.shape[2]
    return torch.reshape(x, (batch_size, extrinsic_dim)).cpu()

def pool_tensor(x, attention_mask):
    masked_reps = x * attention_mask.unsqueeze(-1)
    pooled_rep = masked_reps.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
    return pooled_rep.cpu()

def last_token_rep(x, attention_mask):
    seq_len = attention_mask.sum(dim=1)
    indices = (seq_len - 1)
    last_token_rep = x[torch.arange(x.size(0)), indices]
    return last_token_rep.cpu()

def random_token_rep(x, attention_mask):
    seq_len = attention_mask.sum(dim=1)
    random_idx = np.random.randint(0, seq_len) # returns random int vector in [0, seq_len - 1]
    random_token_rep = x[torch.arange(x.size(0)), (random_idx)]
    return random_token_rep.cpu()

POOLING_METHOD = {
    'last_token': last_token_rep, 'concat': resize_tensor, 'average': pool_tensor, 'random_token': random_token_rep
}
################################################################

def get_representations(model, encodings, pooling_methods=['last_token']):
    print('Getting representations')
    outputs = {pooling_method: [] for pooling_method in pooling_methods}
    with torch.no_grad():
        torch.cuda.empty_cache()
        for batch in tqdm(encodings):
            try:
                output = model(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)['hidden_states']
            except Exception as e:
                print(e)
                print(batch['input_ids'].shape)
                print(batch['attention_mask'].shape)
                continue

            for pooling_method in outputs:
                pooling_fn = POOLING_METHOD[pooling_method]
                pooled_output = tuple([pooling_fn(x, batch['attention_mask']) for x in output])

                outputs[pooling_method].append(pooled_output)

    # Concatenate the batches together
    for pooling_method in outputs:
        representations = outputs[pooling_method]
        representations = [list(batch) for batch in zip(*representations)]
        representations = [torch.cat(batches, dim=0) for batches in representations]
        outputs[pooling_method] = representations
    return outputs

################## ID METHOD #######################
METHODS = {
    'PCA': skdim.id.lPCA(),
    'TwoNN': skdim.id.TwoNN(),
    'FisherS': skdim.id.FisherS(),
    'CorrInt': skdim.id.CorrInt(),
    'MLE': skdim.id.MLE(),
    'TLE': skdim.id.TLE(),
    'ESS': skdim.id.ESS(),
    'MADA': skdim.id.MADA(),
    'MOM': skdim.id.MOM()
}
#####################################################

def compute_layer_ids(representations, method):
    print('Computing layer IDs')
    ids = {method: []}
    for i, representation in tqdm(enumerate(representations)):
        try:
            repr = representation.float() if method == 'ESS' else representation # ESS requires float32 precision
            ids[method].append(METHODS[method].fit_transform(repr))
        except Exception as e:
            print(e)
            # Weird issue where there are many duplicates after the first layer...
            deduped, dup_indices = torch.unique(representation.double(), sorted=False, return_inverse=True, dim=0)

            print('Problem with Layer ', i)
            print('Number of unique representations')
            print(deduped.shape[0])

            ids[method].append(None)

    return ids

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='ID computation')

    # Data selection
    parser.add_argument('--model_name', type=str, default='facebook/opt-6.7b')
    parser.add_argument('--benchmark_name', type=str, default="none", choices=["none", 'glue', 'super_glue', 'bigbench'])
    parser.add_argument('--task_name', type=str, default='bookcorpus',
                        help='GLUE task strings OR e.g.' +
                        'text/optcorpus, optcorpus_permuted_tokens, optcorpus_swapped_tokens,' +
                        'optcorpus_random_tokens')
    parser.add_argument('--subset', type=str, default='Julia-mit', required=False)
    parser.add_argument('--method_name', type=str, default='PCA')
    parser.add_argument('--layer', type=int)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sample_size', type=int, default=50000)
    parser.add_argument('--pooling', nargs='+', default=['last_token'], choices=['concat', 'last_token', 'average', 'random_token'])
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--path_to_results', default='/home/echeng/adapters_id/bootstrap_id_results/')
    parser.add_argument('--path_to_cache', default='/home/echeng/adapters_id/id_results/')
    parser.add_argument('--plot_ids', default=False, type=bool)
    args = parser.parse_args()
    print(args)



    if args.benchmark_name == 'none': args.benchmark_name = ""
    try:
        int(args.task_name)
        args.task_name = bigbench_task_id_to_name[int(args.task_name)]
    except:
        pass

    # Save data
    save_path = args.path_to_results + '{}/{}/pooling_{}/sample_{}'.format(
        args.model_name,
        args.task_name,
        'last_token',
        args.sample_size)

    # Check whether the specified path exists or not
    isExist = os.path.exists(save_path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(save_path)

    # Check if cached reps exist. If not, get them.
    cached_reps_path = args.path_to_cache + '{}/{}/pooling_{}/cached_reps.pt'.format(
        args.model_name,
        args.task_name,
        'last_token')

    path_exists = os.path.exists(cached_reps_path)
    print(cached_reps_path)
    print(path_exists)

    print('reading cached reps')

    # read in cached respresentations
    representations = torch.load(cached_reps_path)[args.layer] # only use last layer

    # Subsample the tensor according to sample size
    indices = torch.randperm(len(representations))[:args.sample_size]
    sample = representations[indices]
    # Get ID for method on the sample.
    id_json = compute_layer_ids([sample], args.method_name)

    id_json[args.method_name] = {args.layer: [
        float(id_value) for id_value in id_json[args.method_name]
    ][0]}
    print('SAVING')
    # Append to jsonl file
    with open(save_path + '/layer_wise_ids.jsonl', 'a') as f:
        json.dump(id_json,f)
        f.write('\n')
