from typing import Optional
from dataclasses import dataclass, field

HF_TOKEN = "your_token_here"
DATASETS_PATH = '/path/to/datasets'

# Batch size for finetuning
get_bs = lambda model_name: 8 if '350' in model_name else (32 if '1.3' in model_name else 16)

# For certain tasks on HF we need to get a certain subset
task_to_config = {
    'wikitext': 'wikitext-103-v1',
    'tweet_eval': 'sentiment',
    'cnn_dailymail': '3.0.0',
}

# Pretty print for plotting
task_to_label = {
    'asi/wikitext_fr': 'wikitext_fr',
    'tweet_eval': 'tweets',
    'NeelNanda/pile-10k': 'pile',
    'AhmedSSoliman/CodeXGLUE-CONCODE': 'code',
    'cnn_dailymail': 'dailymail',
    'stas/openwebtext-10k': 'openwebtext',
    'optcorpus_permuted_tokens': 'optcorpus_permuted',
    'optcorpus_random_tokens': 'optcorpus_random',
    'optcorpus_swapped_tokens': 'optcorpus_swapped',
    'ptb_text_only': 'penntreebank'
}

task_to_keys = {
    'glue':{
        "cola": ("sentence",),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence",),
        "stsb": ("sentence1", "sentence2"),
    },
    'super_glue': {
        'boolq': ('question', 'passage'),
        'multirc': ('paragraph', 'question', 'answer'),
        'wic': ('word', 'sentence1', 'sentence2'),
    },
    '': {
       'imdb': ('text',),
        'bookcorpus': ('text',),
    	'asi/wikitext_fr': ('paragraph',),
        'ptb_text_only': ('sentence',),
        'wikitext': ('text',),
        'tweet_eval': ('text',), # we will use sentiment subset.
        'NeelNanda/pile-10k': ('text',),
        'AhmedSSoliman/CodeXGLUE-CONCODE': ('code',),
        'cnn_dailymail': ('article',),
        'stas/openwebtext-10k': ('text',)
    }
}

my_file = open("path/to/tasks_10k.txt", "r")

# reading the file
tasks_10k = my_file.read()

# replacing end splitting the text
# when newline ('\n') is seen.
tasks_10k = tasks_10k.split("\n")
print(tasks_10k)
my_file.close()

transformations = ['_permuted_tokens', '_swapped_tokens', '_random_tokens']

# Define FINETUNE_TASKS relevant to finetuning and TASKS relevant for the ID vs PPL analysis
FINETUNE_TASKS = list(task_to_keys[''].keys()) + \
        ['wikitext' + n for n in transformations] + \
        ['optcorpus' + n for n in transformations]

TASKS = tasks_10k


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    compute_id: bool = field(
        default=False,
        metadata={"help": "Whether to run the experiment on a lower ID (use id_reduction_factor to set)."}
    )
    id_reduction_factor: Optional[float] = field(
        default=1.0, metadata={"help": "Proportion of model extrinsic dimension to keep. ID = ED * ARG."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
