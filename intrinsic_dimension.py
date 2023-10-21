import torch
import os
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import skdim
from datasets import load_dataset, load_metric, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
import json
import argparse
import pandas as pd
import random
import sys
from arguments import *
from dataset_util import HF_TOKEN, process_dataset, encode_data

def plot_ids(args, ids):
    plt.plot(np.arange(len(ids)) / len(ids), ids, 'o-')
    plt.ylabel('Intrinsic Dimension', fontsize=18)
    plt.xlabel('Relative Layer Depth ({})'.format(args.model_name.split('/')[-1]), fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('{} Representational ID'.format(args.task_name.upper()), fontsize=18)
    plt.tight_layout()
    plt.savefig('{}_id_{}_{}.png'.format(args.task_name, args.model_name.split('/')[-1], args.pooling))

####### REPRESENTATION AGGREGATION METHODS ###############
def resize_tensor(x):
    batch_size = x.shape[0]
    extrinsic_dim = x.shape[1] * x.shape[2]
    return torch.reshape(x, (batch_size, extrinsic_dim)).cpu()

def pool_tensor(x, attention_mask):
    masked_reps = x * attention_mask.unsqueeze(-1)
    pooled_rep = masked_reps.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
    return pooled_rep.cpu()

def last_token_rep(x, attention_mask, padding='right'):
    # print(x)
    # print(attention_mask)
    seq_len = attention_mask.sum(dim=1)
    # print('len of sequences', seq_len)
    indices = (seq_len - 1)
    # print(x.size(0))
    # print(indices)
    last_token_rep = x[torch.arange(x.size(0)), indices] if padding=='right' else x[torch.arange(x.size(0)), -1]
    # print(last_token_rep)
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

def get_representations(model, encodings, padding, pooling_methods=['last_token']):
    print('Getting representations')
    outputs = {pooling_method: [] for pooling_method in pooling_methods}
    with torch.no_grad():
        torch.cuda.empty_cache()
        for batch in tqdm(encodings):

            try:
                # print('input ids: ', batch['input_ids'][0])
                # print('shape: ', batch['input_ids'].shape)
                output = model(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)['hidden_states']
            except Exception as e:
                print(e)
                print(batch['input_ids'].shape)
                print(batch['attention_mask'].shape)
                continue

            for pooling_method in outputs:
                pooling_fn = POOLING_METHOD[pooling_method]
                pooled_output = tuple([pooling_fn(x, batch['attention_mask'], padding=padding) for x in output]) 
                # len-k tuple (each elt is a layer). Each layer is batchsize x embed dim
                
                # print(81)
                # print(len(pooled_output))
                # print(pooled_output[0])
                # print('The token: ', batch['input_ids'][0][batch['attention_mask'].sum(dim=1) - 1])
                # print('The next token: ', batch['input_ids'][0][batch['attention_mask'].sum(dim=1)]) # FINE
                outputs[pooling_method].append(pooled_output)

    # Concatenate the batches together
    for pooling_method in outputs:
        representations = outputs[pooling_method]
        print('Final representations:')
        print(representations)
        representations = [list(batch) for batch in zip(*representations)]
        print('Layer 1 reps: ')
        print(representations[1])
        representations = [torch.cat(batches, dim=0) for batches in representations]
        outputs[pooling_method] = representations
        print('Layer 1 reps shape: ')
        print(representations[1].shape)
        # input()
    return outputs

################## ID METHOD #######################
METHODS = {
    'PCA': skdim.id.lPCA(),
    'TwoNN': skdim.id.TwoNN(),
    'FisherS': skdim.id.FisherS(),
    'CorrInt': skdim.id.CorrInt(),
    # 'DANCo': skdim.id.DANCo(),
    # 'KNN': skdim.id.KNN(),
    'MLE': skdim.id.MLE(),
    'TLE': skdim.id.TLE(),
    'ESS': skdim.id.ESS(),
    'MADA': skdim.id.MADA(),
    # 'MiND_ML': skdim.id.MiND_ML(),
    'MOM': skdim.id.MOM()
}
#####################################################

def compute_layer_ids(representations, method):
    print('Computing layer IDs')
    ids = {method: []}
    for i, representation in tqdm(enumerate(representations)):
        try:
            repr = representation.double() if method == 'ESS' else representation # ESS requires float32 precision
            ids[method].append(METHODS[method].fit_transform(repr))
        except Exception as e:
            print(e)
            # Weird issue where there are many duplicates after the first layer...
            deduped, dup_indices = torch.unique(representation.double(), sorted=False, return_inverse=True, dim=0)

            print('Problem with Layer ', i)
            print('Number of unique representations')
            print(deduped.shape[0])

    #         for i in range(deduped.shape[0]):
    #             # find where dup_indices is equal to i
    #             mask = torch.eq(torch.Tensor([i]), dup_indices).numpy() # len = N
    #             same_representations = [data[i] for i, elt in enumerate(mask) if elt]
    #             if len(same_representations) > 1:
    #                 print('Same representations')
    #                 [print(elt) for elt in same_representations]
    #                 print('\n')
            ids[method].append(None)
    # ids_df = pd.DataFrame(ids)
    # ids_df.index = ids_df.index / len(ids_df)
    return ids #ids_df

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='ID computation')

    # Data selection
    parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
    parser.add_argument('--benchmark_name', type=str, default="none", choices=["none", 'glue', 'super_glue', 'bigbench'])
    parser.add_argument('--task_name', type=str, default='cola',
                        help='GLUE task strings OR e.g.' +
                        'text/optcorpus, optcorpus_permuted_tokens, optcorpus_swapped_tokens,' +
                        'optcorpus_random_tokens')
    parser.add_argument('--subset', type=str, default='Julia-mit', required=False)
    parser.add_argument('--method_name', type=str, default='PCA')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sample_size', type=int, default=50000)
    parser.add_argument('--pooling', nargs='+', default=['last_token', 'average'], choices=['concat', 'last_token', 'average', 'random_token'])
    parser.add_argument('--random_seed', type=int, default=32)
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--path_to_results', default='/home/echeng/adapters_id/id_results/')
    parser.add_argument('--plot_ids', default=False, type=bool)
    args = parser.parse_args()
    print(args)

    random.seed(args.random_seed)

    if args.benchmark_name == 'none': args.benchmark_name = ""
    try:
        int(args.task_name)
        args.task_name = bigbench_task_id_to_name[int(args.task_name)]
    except:
        pass

    # Save data
    save_paths = [args.path_to_results + '{}/{}/pooling_{}/'.format(
        args.model_name,
        args.task_name,
        pooling,
    ) for pooling in args.pooling]

    # Check whether the specified path exists or not
    isExist = [os.path.exists(save_path) for save_path in save_paths]
    for i, exist in enumerate(isExist):
        save_path = save_paths[i]
        if not exist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)

        # Check if we have already computed the ID with this method (ec: added 18/05 because had to restart jobs)
        has_computed_id = False
        if os.path.exists(save_path + '/ids.jsonl'):
            with open(save_path + '/ids.jsonl', 'r') as f:
                json_list = list(f)
                for js in json_list:
                    if args.method_name in js:
                        has_computed_id = True
    print(has_computed_id)
    if has_computed_id: quit()

    # Check if cached reps exist. If not, get them.
    cached_reps_paths = [args.path_to_results + '{}/{}/pooling_{}/cached_reps.pt'.format(
        args.model_name,
        args.task_name,
        pooling,
    ) for pooling in args.pooling
    ]
    paths_exist = [os.path.exists(cached_reps_path) for cached_reps_path in cached_reps_paths]
    print(cached_reps_paths)
    print(paths_exist)
    if not all(paths_exist):
        # Get model, tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=True,
            device_map="auto",
            token=HF_TOKEN
        )
        device = model.device

        ### CAST to 8BIT TO SAVE RAM ###
        for param in model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            use_fast=False,
            token=HF_TOKEN
        )
        if 'Llama-2' in args.model_name:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
 
        model.config.n_positions = model.config.max_position_embeddings if 'bloom' not in args.model_name else 2048 # llama and opt have this param, bloom doesn't

        N, data = process_dataset(args)
        print('N: ', N)
        print(data[0])
        encodings = encode_data(tokenizer, N, data, args.batch_size, model.config.n_positions, device)
        print(encodings[0]['input_ids'].shape)

        # GATHER HIDDEN STATES
        representations = get_representations(model, encodings, tokenizer.padding_side, pooling_methods=args.pooling)
        # dict {pooling method: reps}
        # print('Representation size: ', representations[0].shape)

        # k_layers x Tensor(batch_size x embed_dim)

        # Cache the layers
        for i, pooling_method in enumerate(args.pooling):
            print(representations)
            torch.save(representations[pooling_method], cached_reps_paths[i])
        sys.exit('SAVED CACHED REPS')
    else:
        print('reading cached reps')

        # read in cached respresentations
        assert len(cached_reps_paths) == 1
        representations = torch.load(cached_reps_paths[0])

    print('Len representations: ', len(representations))
    # input()
    print(representations[0].shape)
    # input()

    # Get ID dfs for method in METHOD
    id_json = compute_layer_ids(representations, args.method_name)
    id_json[args.method_name] = [
        str(id_value) for id_value in id_json[args.method_name]
    ]

    # Append to jsonl file
    with open(save_path + '/ids.jsonl', 'a') as f:
        json.dump(id_json,f)
        f.write('\n')
