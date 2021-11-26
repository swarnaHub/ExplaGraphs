import torch
import logging
import os
from enum import Enum
from filelock import FileLock
from typing import List, Optional, Union
import time

from torch.utils.data.dataset import Dataset
from dataclasses import dataclass, field
from transformers.data.processors import DataProcessor
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers.data.datasets import GlueDataTrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.data.processors.glue import glue_convert_examples_to_features
from sklearn.metrics import f1_score
import networkx as nx

logger = logging.getLogger(__name__)


class StanceProcessor(DataProcessor):
    """Processor for the Stance Prediction Task."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")))

    def get_labels(self):
        """See base class."""
        return ["support", "counter"]

    def _create_examples(self, lines):
        """Creates examples for the training, dev and test sets."""
        examples = []
        j = 0
        for (i, line) in enumerate(lines):
            id = i
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=id, text_a=text_a, text_b=text_b, label=label))
        return examples


stance_processor = {
    "stance": StanceProcessor
}

stance_output_modes = {
    "stance": "classification"
}

stance_num_labels = {
    "stance": 2
}


@dataclass
class StanceDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(stance_processor.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class StanceDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
            self,
            args: GlueDataTrainingArguments,
            tokenizer: PreTrainedTokenizer,
            limit_length: Optional[int] = None,
            mode: Union[str, Split] = Split.train,
            cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = stance_processor[args.task_name]()
        self.output_mode = stance_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        self.label_list = self.processor.get_labels()

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=self.label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


def compute_metrics(task_name, output_dir, preds, labels):
    if task_name == "stance":
        with open(os.path.join(output_dir, "gold.txt"), "w", encoding="utf-8-sig") as f:
            for label in labels:
                f.write(str(label) + "\n")
        with open(os.path.join(output_dir, "pred.txt"), "w", encoding="utf-8-sig") as f:
            for pred in preds:
                f.write(str(pred) + "\n")
        return {
            "micro_f1": f1_score(preds, labels, average="micro"),
            "macro_f1": f1_score(preds, labels, average="macro"),
            "weighted_f1": f1_score(preds, labels, average="weighted"),
            "acc": (preds == labels).mean()
            }
    else:
        raise KeyError(task_name)
