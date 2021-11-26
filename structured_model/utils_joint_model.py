from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from collections import OrderedDict
from io import open

import numpy as np
from nltk import word_tokenize

logger = logging.getLogger(__name__)


class ExplaGraphInputExample(object):
    def __init__(self, id, belief, argument, external, node_label_internal_belief, node_label_internal_argument,
                 node_label_external, edge_label, stance_label):
        self.id = id
        self.belief = belief
        self.argument = argument
        self.external = external
        self.node_label_internal_belief = node_label_internal_belief
        self.node_label_internal_argument = node_label_internal_argument
        self.node_label_external = node_label_external
        self.edge_label = edge_label
        self.stance_label = stance_label


class ExplaGraphFeatures(object):
    def __init__(self, id, input_ids, input_mask, segment_ids, node_start_index, node_end_index, node_label,
                 edge_label, stance_label):
        self.id = id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.node_start_index = node_start_index
        self.node_end_index = node_end_index
        self.node_label = node_label
        self.edge_label = edge_label
        self.stance_label = stance_label


class ExplaGraphProcessor(object):
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
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")))

    def get_dev_examples(self, data_dir, is_edge_pred=True):
        # If predicting nodes, then create labels using gold nodes, because don't care
        # But if predicting edges, create node labels using the predicting nodes
        if not is_edge_pred:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), is_eval=True)
        else:
            return self._create_examples_with_predicted_nodes(self._read_tsv(os.path.join(data_dir, "dev.tsv")),
                                                              open(os.path.join(data_dir, "internal_nodes_dev.txt"),
                                                                   "r").read().splitlines(),
                                                              open(os.path.join(data_dir, "external_nodes_dev.txt"),
                                                                   "r").read().splitlines())

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")))

    def get_stance_labels(self):
        return ["support", "counter"]

    def get_node_labels(self):
        return ["B-N", "I-N", "O"]

    def get_edge_labels(self):
        return ["antonym of", "synonym of", "at location", "not at location", "capable of", "not capable of", "causes",
                "not causes", "created by", "not created by", "is a", "is not a", "desires", "not desires",
                "has subevent", "not has subevent", "part of", "not part of", "has context", "not has context",
                "has property", "not has property", "made of", "not made of", "receives action", "not receives action",
                "used for", "not used for", "no relation"]

    def _get_external_nodes_eval(self, belief, argument, external_nodes, internal_nodes_count):
        filtered_external_nodes = []
        for external_node in list(set(external_nodes.split(", "))):
            # We'll consider a maximum of 11 nodes (9+2 shared between belief and argument)
            if internal_nodes_count + len(filtered_external_nodes) == 11:
                break
            if external_node in belief or external_node in argument:
                continue
            words = word_tokenize(external_node)
            if len(words) > 3:
                continue

            filtered_external_nodes.append(external_node)

        return filtered_external_nodes

    def _get_external_nodes_train(self, belief, argument, graph):
        external_nodes = []
        for edge in graph[1:-1].split(")("):
            edge_parts = edge.split("; ")
            if edge_parts[0] not in belief and edge_parts[0] not in argument and edge_parts[0] not in external_nodes:
                external_nodes.append(edge_parts[0])
            if edge_parts[2] not in belief and edge_parts[2] not in argument and edge_parts[2] not in external_nodes:
                external_nodes.append(edge_parts[2])

        return external_nodes

    def _get_internal_nodes(self, belief, argument, graph):
        internal_nodes = {}
        for edge in graph[1:-1].split(")("):
            edge_parts = edge.split("; ")
            if edge_parts[0] in belief or edge_parts[0] in argument:
                length = len(edge_parts[0].split(" "))
                if length not in internal_nodes:
                    internal_nodes[length] = [edge_parts[0]]
                elif edge_parts[0] not in internal_nodes[length]:
                    internal_nodes[length].append(edge_parts[0])
            if edge_parts[2] in belief or edge_parts[2] in argument:
                length = len(edge_parts[2].split(" "))
                if length not in internal_nodes:
                    internal_nodes[length] = [edge_parts[2]]
                elif edge_parts[2] not in internal_nodes[length]:
                    internal_nodes[length].append(edge_parts[2])

        return internal_nodes

    def _get_edge_label(self, node_label_internal_belief, belief, node_label_internal_argument, argument,
                        external_nodes, graph):

        edge_label_map = {label: i for i, label in enumerate(self.get_edge_labels())}

        gold_edges = {}
        for edge in graph[1:-1].split(")("):
            parts = edge.split("; ")
            gold_edges[parts[0], parts[2]] = parts[1]

        ordered_nodes = []
        for i, (word, node_label) in enumerate(zip(belief, node_label_internal_belief)):
            if node_label == "B-N":
                node = word
                if i + 1 < len(belief) and node_label_internal_belief[i + 1] == "I-N":
                    node += " " + belief[i + 1]
                    if i + 2 < len(belief) and node_label_internal_belief[i + 2] == "I-N":
                        node += " " + belief[i + 2]

                ordered_nodes.append(node)

        for i, (word, node_label) in enumerate(zip(argument, node_label_internal_argument)):
            if node_label == "B-N":
                node = word
                if i + 1 < len(argument) and node_label_internal_argument[i + 1] == "I-N":
                    node += " " + argument[i + 1]
                    if i + 2 < len(argument) and node_label_internal_argument[i + 2] == "I-N":
                        node += " " + argument[i + 2]

                ordered_nodes.append(node)

        ordered_nodes.extend(external_nodes)

        edge_label = np.zeros((len(ordered_nodes), len(ordered_nodes)), dtype=int)

        for i in range(len(edge_label)):
            for j in range(len(edge_label)):
                if i == j:
                    edge_label[i][j] = -100
                elif (ordered_nodes[i], ordered_nodes[j]) in gold_edges:
                    edge_label[i][j] = edge_label_map[gold_edges[(ordered_nodes[i], ordered_nodes[j])]]
                else:
                    edge_label[i][j] = edge_label_map["no relation"]

        return list(edge_label.flatten())

    def _get_node_label_internal(self, internal_nodes, words):
        labels = ["O"] * len(words)

        for length in range(3, 0, -1):
            if length not in internal_nodes:
                continue
            nodes = internal_nodes[length]
            for node in nodes:
                node_words = node.split(" ")
                for (i, word) in enumerate(words):
                    if length == 3 and i < len(words) - 2 and words[i] == node_words[0] and words[i + 1] == node_words[
                        1] and words[i + 2] == node_words[2]:
                        if labels[i] == "O" and labels[i + 1] == "O" and labels[i + 2] == "O":
                            labels[i] = "B-N"
                            labels[i + 1] = "I-N"
                            labels[i + 2] = "I-N"
                    if length == 2 and i < len(words) - 1 and words[i] == node_words[0] and words[i + 1] == node_words[
                        1]:
                        if labels[i] == "O" and labels[i + 1] == "O":
                            labels[i] = "B-N"
                            labels[i + 1] = "I-N"
                    if length == 1 and words[i] == node_words[0]:
                        if labels[i] == "O":
                            labels[i] = "B-N"

        return labels

    def _get_node_label_external(self, external_nodes):
        labels = []
        for external_node in external_nodes:
            length = len(word_tokenize(external_node))
            labels.extend(["B-N"] + ["I-N"] * (length - 1))

        return labels

    def _create_examples(self, records, is_eval=False):
        examples = []

        max_edge_length = 0
        for (i, record) in enumerate(records):
            belief = record[0].lower()
            argument = record[1].lower()
            stance_label = record[2]
            graph = record[3].lower()

            belief_words = word_tokenize(belief)
            argument_words = word_tokenize(argument)

            internal_nodes = self._get_internal_nodes(belief, argument, graph)
            node_label_internal_belief = self._get_node_label_internal(internal_nodes, belief_words)
            node_label_internal_argument = self._get_node_label_internal(internal_nodes, argument_words)

            # If evaluating, external nodes are not required for tagging because they will come from generation model
            external_nodes = self._get_external_nodes_train(belief, argument, graph) if not is_eval else []

            node_label_external = self._get_node_label_external(external_nodes)

            edge_label = self._get_edge_label(node_label_internal_belief, belief_words, node_label_internal_argument,
                                              argument_words,
                                              external_nodes, graph)

            max_edge_length = max(max_edge_length, len(edge_label))

            external = []
            for external_node in external_nodes:
                external.extend(word_tokenize(external_node))

            examples.append(
                ExplaGraphInputExample(id=i, belief=belief_words, argument=argument_words, external=external,
                                       node_label_internal_belief=node_label_internal_belief,
                                       node_label_internal_argument=node_label_internal_argument,
                                       node_label_external=node_label_external, edge_label=edge_label,
                                       stance_label=stance_label))

        return examples

    def _get_unique_node_count(self, belief, argument, node_label_internal_belief, node_label_internal_argument):
        nodes = []
        for i, (word, node_label) in enumerate(zip(belief, node_label_internal_belief)):
            if node_label == "B-N":
                node = word
                if i + 1 < len(belief) and node_label_internal_belief[i + 1] == "I-N":
                    node += " " + belief[i + 1]
                    if i + 2 < len(belief) and node_label_internal_belief[i + 2] == "I-N":
                        node += " " + belief[i + 2]

                nodes.append(node)

        for i, (word, node_label) in enumerate(zip(argument, node_label_internal_argument)):
            if node_label == "B-N":
                node = word
                if i + 1 < len(argument) and node_label_internal_argument[i + 1] == "I-N":
                    node += " " + argument[i + 1]
                    if i + 2 < len(argument) and node_label_internal_argument[i + 2] == "I-N":
                        node += " " + argument[i + 2]

                if node not in nodes:
                    nodes.append(node)

        return len(nodes)

    def _create_examples_with_predicted_nodes(self, records, internal_nodes, external_nodes):
        assert len(records) == len(external_nodes)
        examples = []

        sample_breaks = [i for i, x in enumerate(internal_nodes) if x == ""]

        max_node_count = 0
        for (i, record) in enumerate(records):
            belief = record[0].lower()
            argument = record[1].lower()
            stance_label = record[2]

            belief_words = word_tokenize(belief)
            argument_words = word_tokenize(argument)

            start = 0 if i == 0 else sample_breaks[i - 1] + 1
            end = sample_breaks[i]
            belief_lines = internal_nodes[start:(start + len(belief_words))]
            argument_lines = internal_nodes[(start + len(belief_words)):end]

            node_label_internal_belief = [belief_line.split("\t")[1] for belief_line in belief_lines]
            node_label_internal_argument = [argument_line.split("\t")[1] for argument_line in argument_lines]
            node_count = self._get_unique_node_count(belief_words, argument_words, node_label_internal_belief,
                                                     node_label_internal_argument)

            external = []
            node_label_external = []
            for external_node in list(OrderedDict.fromkeys(external_nodes[i].split(", "))):
                # Allowing a maximum of 9 unique nodes, as per the task
                if node_count >= 8:
                    break
                if external_node in belief or external_node in argument:
                    continue
                words = word_tokenize(external_node)
                if len(words) > 3:
                    continue
                node_count += 1
                external.extend(words)
                node_label_external.extend(["B-N"] + ["I-N"] * (len(words) - 1))

            max_node_count = max(max_node_count, node_count)

            edge_label = np.zeros((node_count, node_count), dtype=int)

            for a in range(len(edge_label)):
                for b in range(len(edge_label)):
                    if a == b:
                        edge_label[a][b] = -100
                    else:
                        edge_label[a][b] = 0  # Don't care, some placeholder value

            edge_label = list(edge_label.flatten())

            examples.append(
                ExplaGraphInputExample(id=i, belief=belief_words, argument=argument_words, external=external,
                                       node_label_internal_belief=node_label_internal_belief,
                                       node_label_internal_argument=node_label_internal_argument,
                                       node_label_external=node_label_external, edge_label=edge_label,
                                       stance_label=stance_label))

        print(max_node_count)
        return examples

def get_word_start_indices(examples, tokenizer, cls_token, sep_token):
    all_word_start_indices = []
    for (ex_index, example) in enumerate(examples):
        word_start_indices = []
        print(ex_index)

        tokens = [cls_token]

        for word in example.belief:
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                word_start_indices.append(len(tokens))
                tokens.extend(word_tokens)

        tokens = tokens + [sep_token] + [sep_token]

        for word in example.argument:
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                word_start_indices.append(len(tokens))
                tokens.extend(word_tokens)

        all_word_start_indices.append(word_start_indices)

    return all_word_start_indices


def convert_examples_to_features(examples,
                                 stance_label_list,
                                 node_label_list,
                                 max_seq_length,
                                 max_nodes,
                                 tokenizer,
                                 cls_token='[CLS]',
                                 sep_token='[SEP]'):
    # The encoding is based on RoBERTa (and hence segment ids don't matter)
    node_label_map = {label: i for i, label in enumerate(node_label_list)}
    stance_label_map = {label: i for i, label in enumerate(stance_label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        print(ex_index)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = [cls_token]
        node_label_ids = [-100]
        node_start_index, node_end_index = [], []

        # Encode the belief
        for word, label in zip(example.belief, example.node_label_internal_belief):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                if label == "B-N":
                    node_start_index.append(len(tokens))
                tokens.extend(word_tokens)
                if label == "B-N":
                    node_end_index.append(len(tokens) - 1)
                elif label == "I-N":
                    node_end_index[len(node_end_index) - 1] = len(tokens) - 1  # Update the end index

                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                # node_label_ids.extend([node_label_map[label]] + [-100] * (len(word_tokens) - 1))
                if label == "B-N":
                    node_label_ids.extend(
                        [node_label_map[label]] + [node_label_map["I-N"]] * (len(word_tokens) - 1))
                else:
                    node_label_ids.extend([node_label_map[label]] * len(word_tokens))

        tokens = tokens + [sep_token] + [sep_token]
        node_label_ids = node_label_ids + [-100, -100]

        # Encode the argument
        for word, label in zip(example.argument, example.node_label_internal_argument):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                if label == "B-N":
                    node_start_index.append(len(tokens))
                tokens.extend(word_tokens)
                if label == "B-N":
                    node_end_index.append(len(tokens) - 1)
                elif label == "I-N":
                    node_end_index[len(node_end_index) - 1] = len(tokens) - 1  # Update the end index

                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                # node_label_ids.extend([node_label_map[label]] + [-100] * (len(word_tokens) - 1))
                if label == "B-N":
                    node_label_ids.extend([node_label_map[label]] + [node_label_map["I-N"]] * (len(word_tokens) - 1))
                else:
                    node_label_ids.extend([node_label_map[label]] * len(word_tokens))

        tokens = tokens + [sep_token] + [sep_token]
        node_label_ids = node_label_ids + [-100, -100]

        # Encode the external concepts
        for word, label in zip(example.external, example.node_label_external):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                if label == "B-N":
                    node_start_index.append(len(tokens))
                tokens.extend(word_tokens)
                if label == "B-N":
                    node_end_index.append(len(tokens) - 1)
                elif label == "I-N":
                    node_end_index[len(node_end_index) - 1] = len(tokens) - 1  # Update the end index

                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                # node_label_ids.extend([node_label_map[label]] + [-100] * (len(word_tokens) - 1))
                if label == "B-N":
                    node_label_ids.extend(
                        [node_label_map[label]] + [node_label_map["I-N"]] * (len(word_tokens) - 1))
                else:
                    node_label_ids.extend([node_label_map[label]] * len(word_tokens))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        input_ids = input_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        node_label = node_label_ids + ([-100] * padding_length)
        segment_ids = [0] * len(input_ids)

        padding_length = max_seq_length - len(node_start_index)
        node_start_index = node_start_index + ([0] * padding_length)
        node_end_index = node_end_index + ([0] * padding_length)
        edge_label = example.edge_label + [-100] * (max_nodes * max_nodes - len(example.edge_label))

        stance_label = stance_label_map[example.stance_label]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(node_start_index) == max_seq_length
        assert len(node_end_index) == max_seq_length
        assert len(edge_label) == max_nodes * max_nodes

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("id: %s" % (example.id))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("node_start_index: %s" % " ".join([str(x) for x in node_start_index]))
            logger.info("node_end_index: %s" % " ".join([str(x) for x in node_end_index]))
            logger.info("node_label: %s" % " ".join([str(x) for x in node_label]))
            logger.info("edge_label: %s" % " ".join([str(x) for x in edge_label]))
            logger.info("label: %s (id = %d)" % (example.stance_label, stance_label))

        features.append(
            ExplaGraphFeatures(id=id,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               node_start_index=node_start_index,
                               node_end_index=node_end_index,
                               node_label=node_label,
                               edge_label=edge_label,
                               stance_label=stance_label))

    return features


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "eg":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def write_node_predictions_to_file(writer, test_input_reader, preds_list):
    example_id = 0
    for line in test_input_reader:
        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
            writer.write(line)
            if not preds_list[example_id]:
                example_id += 1
        elif preds_list[example_id]:
            output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
            writer.write(output_line)
        else:
            logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])


processors = {
    "eg": ExplaGraphProcessor
}

output_modes = {
    "eg": "classification"
}
