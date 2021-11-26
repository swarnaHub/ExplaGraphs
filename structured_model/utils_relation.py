from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

logger = logging.getLogger(__name__)


class RelationInputExample(object):
    def __init__(self, id, concept1, concept2, relation):
        self.id = id
        self.concept1 = concept1
        self.concept2 = concept2
        self.relation = relation


class RelationFeatures(object):
    def __init__(self, id, input_ids, input_mask, segment_ids, start_indices, end_indices, relation_label):
        self.id = id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_indices = start_indices
        self.end_indices = end_indices
        self.relation_label = relation_label


class RelationProcessor(object):
    def _read_tsv(self, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "conceptnet_train.tsv")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "conceptnet_test.tsv")))

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")))

    def get_relation_labels(self):
        return ["antonym of", "synonym of", "at location", "not at location", "capable of", "not capable of", "causes",
                "not causes", "created by", "not created by", "is a", "is not a", "desires", "not desires",
                "has subevent", "not has subevent", "part of", "not part of", "has context", "not has context",
                "has property", "not has property", "made of", "not made of", "receives action", "not receives action",
                "used for", "not used for", "no relation"]

    def _create_examples(self, records):
        examples = []
        for (i, record) in enumerate(records):
            concept1 = record[0]
            relation = record[1]
            concept2 = record[2]

            examples.append(
                RelationInputExample(id=i, concept1=concept1, concept2=concept2, relation=relation))

        return examples


def convert_examples_to_features(examples,
                                 relation_label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token='[CLS]',
                                 sep_token='[SEP]'):
    # The encoding is based on RoBERTa (and hence segment ids don't matter)
    relation_label_map = {label: i for i, label in enumerate(relation_label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        print(ex_index)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        concept1 = " ".join(example.concept1.split("_"))
        concept2 = " ".join(example.concept2.split("_"))

        if len(concept1) == 0 or len(concept2) == 0:
            continue

        start_indices, end_indices = [], []
        tokens = [cls_token]
        start_indices.append(len(tokens))

        tokens += tokenizer.tokenize(concept1)
        end_indices.append(len(tokens)-1)

        tokens += [sep_token, sep_token]

        start_indices.append(len(tokens))
        tokens += tokenizer.tokenize(concept2)
        end_indices.append(len(tokens)-1)

        assert start_indices[0] <= end_indices[0]
        assert start_indices[1] <= end_indices[1]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        if len(input_ids) >= max_seq_length:
            continue

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        input_ids = input_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = [0] * len(input_ids)

        relation_label = relation_label_map[example.relation]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("id: %s" % (example.id))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("start_indices: %s" % " ".join([str(x) for x in start_indices]))
            logger.info("end_indices: %s" % " ".join([str(x) for x in end_indices]))
            logger.info("relation label: %s (id = %d)" % (example.relation, relation_label))

        features.append(
            RelationFeatures(id=id,
                             input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             start_indices=start_indices,
                             end_indices=end_indices,
                             relation_label=relation_label))

    return features


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "relation":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


processors = {
    "relation": RelationProcessor
}

output_modes = {
    "relation": "classification"
}
