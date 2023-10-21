import os
from tqdm import tqdm
import json
import random
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from arguments import *
import argparse
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

def encode_data(tokenizer, N, data, batch_size, max_length, device, last_k=None):
    # last_k (int): only use the last k tokens of the input

    # If the input data is text
    if type(data[0]) == str:
        encodings = tokenizer(data, padding=True, truncation=True, max_length=max_length, return_length=True, return_tensors="pt") # output variable length encodings
        if not last_k:
            encodings = [
                {'input_ids': encodings['input_ids'][i: i + batch_size].to(device),
                'attention_mask': encodings['attention_mask'][i: i + batch_size].to(device),
                'length': encodings['length'][i: i + batch_size] }
                for i in range(0, N, batch_size)
            ]
        else:
            encodings = [
                {'input_ids': encodings['input_ids'][i: i + batch_size][-last_k:].to(device),
                'attention_mask': encodings['attention_mask'][i: i + batch_size][-last_k:].to(device) }
                for i in range(0, N, batch_size)
            ]



    else: # input data is tokens-- manually pad and batch.
        max_len = max([len(sentence) for sentence in data])
        data = [sentence for sentence in data if len(sentence) > 2]
        encodings = [tokenizer.encode(sentence[1:], padding='max_length', max_length=max_len, return_tensors="pt") \
                     for sentence in data]
        batched_encodings = [torch.stack(encodings[i: i + batch_size]).squeeze(1).to(device) for i in range(0, len(data), batch_size)]
        batched_attention_masks = [(tokens != 1).to(device).long() for tokens in batched_encodings]
        encodings = [
            {'input_ids': batched_encodings[j], 'attention_mask': batched_attention_masks[j]}
            for j in range(len(batched_encodings))
        ]

    return encodings

def preprocess_regular_dataset(args, context_window_length, tokenizer):
    # Load dataset as list of strings
    N, data = process_dataset(args, remove_empty=True)

    # Encode dataset into tokens (format {'generated_token_ids': }) jsonl
    encoded_data = encode_data(
        tokenizer,
        N,
        data,
        args.batch_size,
        context_window_length,
        device='cpu'
    )
    tokens = torch.concat([batch['input_ids'] for batch in encoded_data])

    # Iterate through
    truncated_data = []
    pad_token_id = tokenizer.pad_token_id

    for tokenized_sentence in tokens.tolist():
        dataline = {}
        PAD_index = tokenized_sentence.index(pad_token_id) if pad_token_id in tokenized_sentence else len(tokenized_sentence)
        generated_token_ids = tokenized_sentence[:PAD_index]
        dataline['generated_token_ids'] = generated_token_ids
        dataline['generated_text'] = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        if dataline['generated_text'] in ('', ' '): continue
        truncated_data.append(dataline)

    # Save to file
    folderpath = f'{DATASETS_PATH}{args.task_name}/{args.model_name}'

    # Save jsonl to file
    filepath = folderpath + f'/{args.task_name.split("/")[-1]}.jsonl'

    # Check whether the specified path exists or not
    if not os.path.exists(folderpath):
        # Create a new directory because it does not exist
        os.makedirs(folderpath)

    # Open json file in append mode
    # And write the results
    with open(filepath, 'w') as f:
        for d in truncated_data:
            json.dump(d, f)
            f.write('\n')

def preprocess_optcorpus(model_name, dataset_name, context_window_length, tokenizer):
    """Preprocesses the corpus so that all of the datapoints are within the context window of the model
    model_name = opt-350m, opt-1.3b, opt-6.7b
    """
    folderpath = f'{DATASETS_PATH}{model_name}'
    filepaths = os.scandir(folderpath)

    for filepath in tqdm(filepaths):
        # Read in file
        with open(folderpath + '/' + filepath.name, 'r') as json_file:
            json_list = list(json_file)
            data = list(filter(lambda x: len(x), [json.loads(json_str) for json_str in json_list]))

        # if the length of the tokens greater than the context window length, split the datapoint and remake the text
        truncated_data = []
        for dataline in data:
            # Find the EOS
            generated_token_ids = dataline['generated_token_ids']
            EOS_index = generated_token_ids[1:].index(2) + 1 # ignore the BOS token (also 2) which may start the sentence
            generated_token_ids = generated_token_ids[:EOS_index + 1] # cut off everything after the EOS index.

            if len(generated_token_ids) <= context_window_length:
                dataline['generated_token_ids'] = generated_token_ids
                text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                dataline['generated_text'] = text
                truncated_data.append(dataline)
            else:
                # split the datapoint
                token_ids = generated_token_ids
                truncation_points = list(range(0, len(token_ids), context_window_length)) + [len(token_ids)]
                token_ids_truncated = [generated_token_ids[truncation_points[i]: truncation_points[i+1]] for i in range(len(truncation_points) - 1)]

                # check that we kept all the tokens
                assert len(token_ids) == sum([len(chunk) for chunk in token_ids_truncated])

                # now remap tokens to text
                texts_truncated = [tokenizer.decode(token_ids_chunk, skip_special_tokens=True) for token_ids_chunk in token_ids_truncated]

                for i, token_id_chunk in enumerate(token_ids_truncated):
                    truncated_data.append({'generated_text': texts_truncated[i], 'generated_token_ids': token_id_chunk})

        new_filepath = filepath.name
        new_folderpath = f'{DATASETS_PATH}{dataset_name}/{model_name}'

        # Check whether the specified path exists or not
        if not os.path.exists(new_folderpath):
            # Create a new directory because it does not exist
            os.makedirs(new_folderpath)

        # Open json file in append mode
        # And write the results
        with open(new_folderpath + '/' + new_filepath, 'w') as f:
            for d in truncated_data:
                json.dump(d,f)
                f.write('\n')

def permute_tokens_optcorpus(model_name, dataset_name, vocab_size, tokenizer):
    """Deterministically maps each token to another in optcorpus.
    """
    folderpath = f'{DATASETS_PATH}{dataset_name}/{model_name}'
    filepaths = os.scandir(folderpath)

    # Make deterministic mapping for non-special tokens
    deterministic_mapping = list(range(vocab_size))
    special_tokens_id = 3 # for opt the specials are 0 1 2
    assignment = list(range(special_tokens_id, vocab_size))
    random.shuffle(assignment)
    deterministic_mapping[special_tokens_id:] = assignment

    for filepath in tqdm(filepaths):
        if 'swapped_tokens' in str(filepath) or 'permuted_tokens' in str(filepath) or '.jsonl' not in str(filepath): continue

        # Read in file
        with open(folderpath + '/' + filepath.name, 'r') as json_file:
            json_list = list(json_file)
            data = [json.loads(json_str)['generated_token_ids'] for json_str in json_list]

        swapped_data = []
        for dataline in data:
            dataline = [deterministic_mapping[token] for token in dataline]
            try:
                swapped_data.append({'generated_text': tokenizer.decode(dataline, skip_special_tokens=True), 'generated_token_ids': dataline})
            except Exception as e:
                print(e)

        new_filepath = filepath.name
        new_folderpath = f'{DATASETS_PATH}{dataset_name}/{model_name}/swapped_tokens/'

        # Check whether the specified path exists or not
        if not os.path.exists(new_folderpath):
            # Create a new directory because it does not exist
            os.makedirs(new_folderpath)

        # Open json file in append mode
        # And write the results
        with open(new_folderpath + new_filepath, 'w') as f:
            for d in swapped_data:
                json.dump(d,f)
                f.write('\n')

def random_tokens_optcorpus(model_name, dataset_name, vocab_size, tokenizer):
    """Deterministically maps each token to another in optcorpus.
    """
    folderpath = f'{DATASETS_PATH}{dataset_name}/{model_name}'
    filepaths = os.scandir(folderpath)

    # exclude starting with token id
    special_tok_ids = 3 # 0, 1, 2 are specials for opt
    candidate_tokens = range(special_tok_ids, vocab_size)

    for filepath in tqdm(filepaths):
        if '_tokens' in str(filepath) or '.jsonl' not in str(filepath): continue
        # Read in file
        with open(folderpath + '/' + filepath.name, 'r') as json_file:
            json_list = list(json_file)
            data = [json.loads(json_str)['generated_token_ids'] for json_str in json_list]

        swapped_data = []
        for dataline in data:
            dataline = [random.choice(candidate_tokens) if token >= special_tok_ids else token for token in dataline]
            swapped_data.append({'generated_text': tokenizer.decode(dataline, skip_special_tokens=True), 'generated_token_ids': dataline})

        new_filepath = filepath.name
        new_folderpath = f'{DATASETS_PATH}{dataset_name}/{model_name}/random_tokens/'

        # Check whether the specified path exists or not
        if not os.path.exists(new_folderpath):
            # Create a new directory because it does not exist
            os.makedirs(new_folderpath)

        # Open json file in append mode
        # And write the results
        with open(new_folderpath + new_filepath, 'w') as f:
            for d in swapped_data:
                json.dump(d,f)
                f.write('\n')


def permute_within_string_optcorpus(model_name, dataset_name, tokenizer):
    """Permutes the tokens within each datapoint in optcorpus.
    dataset_name: opt-corpus-truncated, wikitext, CodeXGLUE-CONCODE
    """
    folderpath = f'{DATASETS_PATH}{dataset_name}/{model_name}'
    filepaths = os.scandir(folderpath)

    for filepath in tqdm(filepaths):
        if 'permuted_tokens' in str(filepath) or '.jsonl' not in str(filepath): continue
        # Read in file
        with open(folderpath + '/' + filepath.name, 'r') as json_file:
            json_list = list(json_file)
            data = [json.loads(json_str)['generated_token_ids'] for json_str in json_list]

        # if the start token is 2 (which should usually be the case), permute the rest
        start_tok = end_tok = 2
        permuted_data = []
        for dataline in data:
            if end_tok in dataline[1:-1]:
                print('PROBLEM with DATALINE: EOS in the middle.')
                print(dataline.index(end_tok))
                print(dataline)
            if int(dataline[0]) == start_tok:
                print('Shuffling all middle tokens')
                permutation_result = dataline[1:-1]
                random.shuffle(permutation_result)
                dataline[1:-1] = permutation_result
            else:
                # shuffle all but last token
                print('Shuffle all but last token (it was truncated)')
                permutation_result = dataline[:-1]
                random.shuffle(permutation_result)
                dataline[:-1] = permutation_result
            # print(type(dataline[0]))
            assert start_tok not in dataline[1:-1]
            permuted_data.append({'generated_text': tokenizer.decode(dataline, skip_special_tokens=True), 'generated_token_ids': dataline})

        new_filepath = filepath.name
        new_folderpath = f'{DATASETS_PATH}{dataset_name}/{model_name}/permuted_tokens/'

        # Check whether the specified path exists or not
        if not os.path.exists(new_folderpath):
            # Create a new directory because it does not exist
            os.makedirs(new_folderpath)

        # Open json file in append mode
        # And write the results
        with open(new_folderpath + new_filepath, 'w') as f:
            for d in permuted_data:
                json.dump(d,f)
                f.write('\n')

def read_local_dataset(task_name, model_name):
    """Reads local dataset in /datasets/COLT

    Args:
        task_name: {mode}/{task_name}, mode in [text, token], task_name in {task_name}_{swapped/permuted/random}_tokens
        model_name (_type_): _description_
    """
    mode = task_name[:task_name.index('/')]
    rest = task_name[task_name.index('/'):]
    if 'wikitext' in rest:
        task_name, transformation = 'wikitext', rest[len('wikitext_')+1:]

    folderpath = f'{DATASETS_PATH}{task_name}/{model_name}/{transformation}'
    filepaths = [f.name for f in os.scandir(folderpath) if f.name.endswith('.jsonl')]

    print(filepaths)
    data = []
    for filepath in filepaths:
        with open(folderpath + '/' + filepath, 'r') as json_file:
            json_list = list(json_file)
            data.extend(list(filter(lambda x: len(x), [json.loads(json_str)['generated_token_ids' if mode=='tokens' else 'generated_text'] for json_str in json_list])))

    return data


def read_optcorpus(task_name, model_name):
    """Reads optcorpus folder

    Args:
        task_name: {mode}/{task_name}, mode in [text, token], task_name in [optcorpus, optcorpus_{swapped/permuted/random}_tokens]
    """
    mode, task_name = task_name.split('/')
    model_name = model_name.split('/')[-1]
    folderpath = f'{DATASETS_PATH}{model_name.split("-")[0]}-corpus-truncated/{model_name}/'
    if task_name=='optcorpus': folderpath += '_'.join(task_name.split('_')[1:])
    filepaths = [f.name for f in os.scandir(folderpath) if f.name.endswith('.jsonl') ]

    # Just the lists of text
    data = []
    for filepath in filepaths:
        with open(folderpath + '/' + filepath, 'r') as json_file:
            json_list = list(json_file)
            data.extend(list(filter(lambda x: len(x), [json.loads(json_str)['generated_token_ids' if mode=='tokens' else 'generated_text'] for json_str in json_list])))

    return data

def streaming_load_dataset(args, split='train'):
    """Sometimes the datasets are too big to fit in memory, so we load the first args.sample_size
    examples via streaming.

    Args:
        args (NameSpace):

    """
    ds = load_dataset(args.task_name, args.subset, split=split, streaming=True)
    iter_ds = iter(ds)

    data = []
    for _ in range(args.sample_size):
        next_data = next(iter_ds)
        if next_data:
            sentence_keys = task_to_keys[args.benchmark_name][args.task_name]
            for sentence_key in sentence_keys:
                data.append(next_data[sentence_key])
        else:
            break
    return len(data), list(set(data)) # shuffle data


def process_dataset(args, remove_empty=False):
    """Given a GLUE task name, returns the dataset as a list of inputs. When the inputs are two sentences, processes them separately.

    Args:
        task_name (str): _description_
        args (str): _description_

    Returns:
        _type_: _description_
    """
    if 'corpus' in args.task_name:
        data = read_optcorpus(args.task_name, args.model_name)
    elif 'permuted' in args.task_name or 'random' in args.task_name or 'swapped' in args.task_name:
        # task_name = e.g. tokens/wikitext_permuted_tokens
        data = read_local_dataset(args.task_name, args.model_name)
    else:
        # BENCHMARK TASKS
        if args.benchmark_name not in ("", None):
            data = load_dataset(args.benchmark_name, args.task_name)
        else:
            # imdb, math, etc
            if args.task_name in task_to_config:
                data = load_dataset(args.task_name, task_to_config[args.task_name])
            else:
                data = load_dataset(args.task_name)#, download_mode="force_redownload")

        data = concatenate_datasets([data[split] for split in data])

        # Format the GLUE tasks into a list concatenating sentence 1s and sentence 2s
        if args.benchmark_name in task_to_keys:
            sentence_keys = task_to_keys[args.benchmark_name][args.task_name] # list of sentence keys
        else:
            sentence_keys = ['text']

        data = [datum for data_part in [data[sentence_key] for sentence_key in sentence_keys] for datum in data_part]

    # Dedup and remove empty inputs
    if type(data[0]) is str: # only if data contains text, and not tokens
        data = list(set(data))
        if remove_empty:
            data = [d for d in data if (d!="" and d!=" ")]
    else: # tokens
        if remove_empty:
            data = [d for d in data if (d!=[2, 2] and d!=[2,])]


    if args.sample_size:
        N = min(args.sample_size, len(data))
        data = data[:N]
    else:
        N = len(data)
    return N, data

def jsonl_to_csv(folderpath):
    """For a folderpath containing a .jsonl file, converts it to id_df csv

    Args:
        folderpath (str): path to folder
    """
    pass

def process_finetune_output(task_list, model_list):
    for task in task_list:
        for model_name in model_list:
            batch_size = get_bs(model_name)
            path = f'/home/echeng/adapters_id/finetune_output/{task}/{model_name}/bs_{batch_size}'
            best_lr_loss = - np.inf
            best_lr = None

            for lr in ['0.00005', '0.000005', '0.0000005']:
                lr_path = path + f'/lr_{lr}'
                run_summaries = []

                # Average + std results across random seeds, save a summary json.
                for rs in [0, 1, 2]:
                    path_rs = lr_path + f'/rs_{rs}'
                    if not os.path.exists(path_rs):
                        print(f"task {task} lr {lr} rs {rs} does not exist!")
                        continue
                    files_in_folder = os.listdir(path_rs)
                    if 'trainer_state.json' in files_in_folder:
                        last_checkpoint_path = path_rs
                    else:
                        checkpoint_dirs = [f for f in os.listdir(path_rs) if 'checkpoint' in f]
                        if not len(checkpoint_dirs):
                            print(f'task {task} lr {lr} RS {rs} has no checkpoint!!')
                            continue

                        last_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))[-1]

                        # Get the trainer state and compute statistics
                        last_checkpoint_path = path_rs + f'/{last_checkpoint}'

                    trainer_state = last_checkpoint_path + '/trainer_state.json'
                    print('trainer state path: ', trainer_state)
                    with open(trainer_state, 'r') as f:
                        run = json.load(f)

                    single_run_summary = {}

                    # Final eval loss
                    single_run_summary['final_eval_loss'] = run['best_metric'] if run['best_metric'] is not None else np.nan
                    print(run['best_metric'])
                    single_run_summary['final_eval_ppl'] = np.exp(run['best_metric']) if run['best_metric'] is not None else np.nan

                    # Number of training epochs
                    single_run_summary['epochs'] = run['epoch']

                    # Number of training steps
                    single_run_summary['steps'] = run['global_step']

                    log_history = run['log_history']

                    # Eval loss history as list
                    single_run_summary['eval_loss_history'] = [step['eval_loss'] for step in log_history if 'eval_loss' in step]
                    single_run_summary['eval_ppl_history'] = [np.exp(eval_loss) for eval_loss in single_run_summary['eval_loss_history']]

                    # Step as list
                    single_run_summary['step_history'] = [step['step'] for step in log_history if 'eval_loss' in step]
                    single_run_summary['epoch_history'] = [step['epoch'] for step in log_history if 'eval_loss' in step]

                    # Speed of learning (trapezoid method) for 1000 steps
                    single_run_summary['speed_of_learning_loss'] = list(np.cumsum(single_run_summary['eval_loss_history']))
                    single_run_summary['speed_of_learning_ppl'] = list(np.cumsum(single_run_summary['eval_ppl_history']))

                    # Sample complexity as 1/T * sum(loss)
                    single_run_summary['sample_complexity_loss'] = np.sum(single_run_summary['eval_loss_history']) / len(single_run_summary['eval_loss_history'])
                    single_run_summary['sample_complexity_ppl'] = np.sum(single_run_summary['eval_ppl_history']) / len(single_run_summary['eval_ppl_history'])

                    # Save new json
                    with open(last_checkpoint_path + '/single_run_summary.json', 'w') as f:
                        json.dump(single_run_summary, f)

                    run_summaries.append(single_run_summary)


                # average and standard deviation each metric
                summary = {}
                longest_run_length = max([len(rs['step_history']) for rs in run_summaries])

                for metric in run_summaries[0]:
                    if type(run_summaries[0][metric]) != list:
                        metric_list_over_seeds = [rs[metric] for rs in run_summaries]
                        summary[metric] = np.nanmean(metric_list_over_seeds)
                        summary[metric + '_std'] = np.nanstd(metric_list_over_seeds)

                    elif type(run_summaries[0][metric]) == list:
                        # for eval loss history, average and standard deviation
                        # construct np array up to longest_run_length
                        runs = [np.ones((longest_run_length,)) * np.nan for _ in run_summaries]

                        # fill in the data
                        for i, rs in enumerate(run_summaries): runs[i][:len(rs[metric])] = rs[metric]

                        summary[metric] = list(np.nanmean(runs, axis=0))
                        assert len(summary[metric]) == longest_run_length

                        if 'step' not in metric and 'epoch' not in metric:
                            summary[metric + '_std'] = list(np.nanstd(runs, axis=0))

                # Save summary json into the lr folder
                with open(lr_path + '/summary.json', 'w') as f:
                    json.dump(summary, f)

                if summary['final_eval_loss'] > best_lr_loss:
                    best_summary = summary
                    best_lr_loss = summary['final_eval_loss']
                    best_lr = lr

            # Pick the lowest eval accuracy one and copy that json to the path folder.
            best_summary['lr'] = best_lr
            with open(path + '/best_lr_summary.json', 'w') as f:
                json.dump(best_summary, f)


def make_ensemble():
    for task in tasks_10k:
        path = f'/path/to/id_results/{model_name}/{task}/pooling_{pooling}/id_df.csv'
        id_df = pd.read_csv(path)
        id_df = id_df.replace('None',np.nan, regex=True)
        id_df = id_df.fillna(value=np.nan)

        id_df = id_df.astype('float64')
        id_metrics = list(set(id_df.columns).intersection(set(id_methods)))
        for metric in id_metrics:
            id_df[metric] = id_df[metric].astype('float64')

        id_df['Ensemble'] = id_df[id_metrics].median(axis=1, skipna=True, numeric_only=True)
        id_df.to_csv(path)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='ID computation')

    # Data selection
    parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
    parser.add_argument('--benchmark_name', type=str, default="", choices=["none", 'glue', 'super_glue', 'bigbench'])
    parser.add_argument('--task_name', type=str, default='cola',
                        help='GLUE task strings OR e.g.' +
                        'text/optcorpus, optcorpus_permuted_tokens, optcorpus_swapped_tokens,' +
                        'optcorpus_random_tokens')
    parser.add_argument('--subset', type=str, default='Julia-mit', required=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sample_size', type=int, default=50000)
    args = parser.parse_args()
    print(args)

    # Get model, tokenizer
    model_name = args.model_name
    id_methods = ['ESS', 'MLE', 'TLE', 'MADA', 'MOM', 'PCA', 'FisherS', 'CorrInt']
    model = AutoModelForCausalLM.from_pretrained(
         model_name,
         load_in_8bit=True,
         device_map="auto",
         token=HF_TOKEN
     )
    device = model.device

    # # ### CAST to 8BIT TO SAVE RAM ###
    for param in model.parameters():
         param.requires_grad = False  # freeze the model - train adapters later
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if 'Llama-2' in args.model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    model.config.n_positions = model.config.max_position_embeddings

    ###################################################################################################
    # Process optcorpus
    name = model_name.split('/')[-1]
    preprocess_regular_dataset(args, model.config.n_positions - 1 if 'opt' in args.model_name else model.config.n_positions, tokenizer)
    preprocess_optcorpus(model_name, 'opt-corpus-truncated', model.config.n_positions - 1, tokenizer) # subtract 1 bc the tokenizer always adds the </s>
    permute_within_string_optcorpus(model_name, args.task_name, tokenizer)
    permute_tokens_optcorpus(model_name, args.task_name, tokenizer.vocab_size, tokenizer)
    random_tokens_optcorpus(model_name, args.task_name, tokenizer.vocab_size, tokenizer)

    ###################################################################################################
    # Create id summaries
    pooling='last_token'
    for task in ['tokens/wikitext_permuted_tokens', 'tokens/wikitext_random_tokens', 'tokens/wikitext_swapped_tokens']
        data = {}
        try:
            path = f'/path/to/id_results/{model_name}/{task}/pooling_{pooling}'
            with open(path + '/ids.jsonl', 'r') as f:
                json_list = list(f)
                for json_str in json_list:
                    data.update(json.loads(json_str))
            data = pd.DataFrame(data)
            data.index = data.index / len(data.index)
            data.to_csv(path + '/id_df.csv')
        except:
            print(task)
            input()

    ########################################################################################3##
    # make finetune summary jsons
    process_finetune_output(FINETUNE_TASKS, ['facebook/opt-350m'])

