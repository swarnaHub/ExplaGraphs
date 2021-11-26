from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import logging
import os
import random
import math

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from scipy.special import softmax
import pathlib

from pytorch_transformers import (WEIGHTS_NAME, RobertaConfig, RobertaTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from joint_model import RobertaForEX
from utils_joint_model import (compute_metrics, output_modes, processors, convert_examples_to_features, get_word_start_indices)
from inference import solve_LP, solve_LP_no_connectivity

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'roberta_eg': (RobertaConfig, RobertaForEX, RobertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    processor = processors[args.task_name]()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_pct is None:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=math.floor(args.warmup_pct * t_total), t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    # set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0],
                              mininterval=10, ncols=100)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet', 'bert_mc'] else None,
                      # XLM don't use segment_ids
                      'node_start_index': batch[3],
                      'node_end_index': batch[4],
                      'node_label': batch[5],
                      'edge_label': batch[6],
                      'stance_label': batch[7]}
            outputs = model(**inputs)
            loss, node_loss, edge_loss = outputs[:3]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                '''
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, processor, eval_split="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                '''

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        # evaluate(args, model, tokenizer, processor, prefix=global_step, eval_split="dev")

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def get_ordered_nodes(dev_examples):
    all_ordered_nodes = []

    for dev_example in dev_examples:
        belief = dev_example.belief
        node_label_internal_belief = dev_example.node_label_internal_belief
        argument = dev_example.argument
        node_label_internal_argument = dev_example.node_label_internal_argument
        external = dev_example.external
        node_label_external = dev_example.node_label_external

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

        for i, (word, node_label) in enumerate(zip(external, node_label_external)):
            if node_label == "B-N":
                node = word
                if i + 1 < len(external) and node_label_external[i + 1] == "I-N":
                    node += " " + external[i + 1]
                    if i + 2 < len(external) and node_label_external[i + 2] == "I-N":
                        node += " " + external[i + 2]

                ordered_nodes.append(node)

        all_ordered_nodes.append(ordered_nodes)

    return all_ordered_nodes


def evaluate(args, model, tokenizer, processor, prefix="", eval_split=None):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    assert eval_split is not None

    results = {}
    if os.path.exists("/output/metrics.json"):
        with open("/output/metrics.json", "r") as f:
            existing_results = json.loads(f.read())
        f.close()
        results.update(existing_results)

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, examples = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True,
                                                         eval_split=eval_split)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} on {} *****".format(prefix, eval_split))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        node_preds = None
        edge_preds = None
        out_node_label_ids = None
        out_edge_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating", mininterval=10, ncols=100):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet', 'bert_mc'] else None,
                          # XLM don't use segment_ids
                          'node_start_index': batch[3],
                          'node_end_index': batch[4],
                          'node_label': batch[5],
                          'edge_label': batch[6],
                          'stance_label': batch[7]}
                outputs = model(**inputs)
                tmp_eval_loss, tmp_node_loss, tmp_edge_loss, node_logits, edge_logits = outputs[:5]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if node_preds is None:
                node_preds = node_logits.detach().cpu().numpy()
                edge_preds = edge_logits.detach().cpu().numpy()
                if not eval_split == "test":
                    out_node_label_ids = inputs['node_label'].detach().cpu().numpy()
                    out_edge_label_ids = inputs['edge_label'].detach().cpu().numpy()
            else:
                node_preds = np.append(node_preds, node_logits.detach().cpu().numpy(), axis=0)
                edge_preds = np.append(edge_preds, edge_logits.detach().cpu().numpy(), axis=0)
                if not eval_split == "test":
                    out_node_label_ids = np.append(out_node_label_ids,
                                                   inputs['node_label'].detach().cpu().numpy(), axis=0)
                    out_edge_label_ids = np.append(out_edge_label_ids,
                                                   inputs['edge_label'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        # The model outputs the node predictions and the edge logit predictions
        # For predicting the graphs, we use ILP inference as specified in README

        # Node Predictions
        if not args.do_eval_edge:
            node_preds = np.argmax(node_preds, axis=2)
            batch_size, seq_len = node_preds.shape
            preds_list = [[] for _ in range(batch_size)]

            node_labels = processor.get_node_labels()
            node_label_map = {i: label for i, label in enumerate(node_labels)}

            dev_examples = processor.get_dev_examples(args.data_dir, args.do_eval_edge)
            all_word_start_indices = get_word_start_indices(dev_examples, tokenizer, tokenizer.cls_token, tokenizer.sep_token)

            for i in range(batch_size):
                for j in all_word_start_indices[i]:
                    preds_list[i].append(node_label_map[node_preds[i][j]])

            # Save Node Predictions
            output_node_predictions_file = os.path.join(eval_output_dir, "prediction_nodes_{}.lst".format(eval_split))
            with open(output_node_predictions_file, "w") as writer:
                assert len(dev_examples) == len(preds_list)
                for (i, pred_sample) in enumerate(preds_list):
                    print(i)
                    words = dev_examples[i].belief + dev_examples[i].argument + dev_examples[i].external
                    assert len(words) == len(pred_sample)
                    prev_pred = "O"
                    for j, (word, pred) in enumerate(zip(words, pred_sample)):
                        # Post-process to take valid nodes (If an I-N occurs without B-N, consider it O)
                        if pred == "O":
                            writer.write(word + "\t" + "O" + "\n")
                            prev_pred = "O"
                        elif pred == "I-N":
                            if prev_pred == "O":
                                writer.write(word + "\t" + "O" + "\n")
                                prev_pred = "O"
                            else:
                                end = j
                                if end - start > 2:
                                    writer.write(word + "\t" + "O" + "\n")
                                    prev_pred = "O"
                                else:
                                    writer.write(word + "\t" + pred + "\n")
                                    prev_pred = pred
                        else:
                            start, end = j, j
                            writer.write(word + "\t" + pred + "\n")
                            prev_pred = pred

                    writer.write("\n")

        # Save edge predictions (This should only be executed after the nodes have been saved)
        if args.do_eval_edge:
            edge_probs = softmax(edge_preds, axis=2)
            no_edge_probs = edge_probs[:, :, -1]
            edge_presence_probs = edge_probs[:, :, :-1]
            max_edge_indices = np.argmax(edge_presence_probs, axis=2)
            max_edge_probs = np.max(edge_presence_probs, axis=2)

            dev_examples = processor.get_dev_examples(args.data_dir, args.do_eval_edge)
            all_ordered_nodes = get_ordered_nodes(dev_examples)
            edge_label_list = processor.get_edge_labels()

            assert len(dev_examples) == len(all_ordered_nodes)
            output_edge_predictions_file = os.path.join(eval_output_dir, "prediction_edges_{}.lst".format(eval_split))
            with open(output_edge_predictions_file, "w") as writer:
                assert len(dev_examples) == len(edge_preds)

                for t, (dev_example, edge_gold, no_edge_prob, max_edge_prob, max_edge_index) in enumerate(
                        zip(dev_examples, out_edge_label_ids, no_edge_probs, max_edge_probs, max_edge_indices)):
                    print(t)
                    num_nodes = dev_example.node_label_internal_belief.count("B-N") \
                                + dev_example.node_label_internal_argument.count("B-N") \
                                + dev_example.node_label_external.count("B-N")
                    no_edge_prob = np.array(no_edge_prob[:(num_nodes * num_nodes)]).reshape(num_nodes, num_nodes)
                    max_edge_prob = np.array(max_edge_prob[:(num_nodes * num_nodes)]).reshape(num_nodes, num_nodes)
                    max_edge_index = np.array(max_edge_index[:(num_nodes * num_nodes)]).reshape(num_nodes, num_nodes)

                    assert len(all_ordered_nodes[t]) == len(no_edge_prob)
                    assert len(all_ordered_nodes[t]) == len(max_edge_prob)
                    edges = solve_LP(no_edge_prob, max_edge_prob, max_edge_index, all_ordered_nodes[t], edge_label_list)
                    writer.write("".join(edges) + "\n")

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, eval_split="train"):
    processor = processors[task]()
    # Load data features from cache or dataset file
    if args.data_cache_dir is None:
        data_cache_dir = args.data_dir
    else:
        data_cache_dir = args.data_cache_dir

    cached_features_file = os.path.join(data_cache_dir, 'cached_{}_{}_{}_{}'.format(
        eval_split,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if eval_split == "dev":
            examples = processor.get_dev_examples(args.data_dir, args.do_eval_edge)
        else:
            examples = None
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        if eval_split == "train":
            examples = processor.get_train_examples(args.data_dir)
        elif eval_split == "dev":
            examples = processor.get_dev_examples(args.data_dir, args.do_eval_edge)
        elif eval_split == "test":
            examples = processor.get_test_examples(args.data_dir)
        else:
            raise Exception("eval_split should be among train / dev / test")

        features = convert_examples_to_features(examples, processor.get_stance_labels(), processor.get_node_labels(),
                                                args.max_seq_length, args.max_nodes,
                                                tokenizer, cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_node_start_index = torch.tensor([f.node_start_index for f in features], dtype=torch.long)
    all_node_end_index = torch.tensor([f.node_end_index for f in features], dtype=torch.long)
    all_node_label = torch.tensor([f.node_label for f in features], dtype=torch.long)
    all_edge_label = torch.tensor([f.edge_label for f in features], dtype=torch.long)
    all_stance_label = torch.tensor([f.stance_label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_node_start_index, all_node_end_index,
                            all_node_label, all_edge_label, all_stance_label)
    return dataset, examples


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: RoBERTaConfig")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--data_cache_dir", default=None, type=str,
                        help="Cache dir if it needs to be diff from data_dir")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=300, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_nodes", default=11, type=int,
                        help="Maximum number of nodes")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_edge", action='store_true',
                        help="Whether to run eval for edges.")
    parser.add_argument("--do_prediction", action='store_true',
                        help="Whether to run prediction on the test set. (Training will not be executed.)")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--run_on_test', action='store_true')

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_pct", default=None, type=float,
                        help="Linear warmup over warmup_pct*total_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_stance_labels()
    num_labels_stance = len(label_list)

    num_labels_node = len(processor.get_node_labels())
    num_labels_edge = len(processor.get_edge_labels())

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels_stance,
        finetuning_task=args.task_name
    )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Prediction (on test set)
    if args.do_prediction:
        results = {}
        logger.info("Prediction on the test set (note: Training will not be executed.) ")
        result = evaluate(args, model, tokenizer, processor, prefix="", eval_split="test")
        result = dict((k, v) for k, v in result.items())
        results.update(result)
        logger.info("***** Experiment finished *****")
        return results

    # Training
    if args.do_train:
        train_dataset, _ = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    checkpoints = [args.output_dir]
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, processor, prefix=global_step, eval_split="dev")
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    # Run on test
    if args.run_on_test and args.local_rank in [-1, 0]:
        checkpoint = checkpoints[0]
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, processor, prefix=global_step, eval_split="test")
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)

    logger.info("***** Experiment finished *****")
    return results


if __name__ == "__main__":
    main()