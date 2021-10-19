import argparse
import networkx as nx
import numpy as np
from graph_matching import split_to_edges, get_tokens, get_bleu_rouge, get_bert_score, get_ged


def is_edge_count_correct(edges):
    if len(edges) < 3:
        return False
    else:
        return True


def is_graph(edges):
    for edge in edges:
        components = edge.split("; ")
        if len(components) != 3:
            return False

    return True


def is_edge_structure_correct(edges, relations):
    for edge in edges:
        components = edge.split("; ")
        if components[0] == "" or len(components[0].split(" ")) > 3:
            return False
        if components[1] not in relations:
            return False
        if components[2] == "" or len(components[2].split(" ")) > 3:
            return False

    return True


def two_concepts_belief_argument(edges, belief, argument):
    belief_concepts = {}
    argument_concepts = {}
    for edge in edges:
        components = edge.split("; ")
        if components[0] in belief:
            belief_concepts[components[0]] = True

        if components[2] in belief:
            belief_concepts[components[2]] = True

        if components[0] in argument:
            argument_concepts[components[0]] = True

        if components[2] in argument:
            argument_concepts[components[2]] = True

    if len(belief_concepts) < 2 or len(argument_concepts) < 2:
        return False
    else:
        return True


def is_connected_DAG(edges):
    g = nx.DiGraph()
    for edge in edges:
        components = edge.split("; ")
        g.add_edge(components[0], components[2])

    return nx.is_weakly_connected(g) and nx.is_directed_acyclic_graph(g)


def get_max(first_precisions, first_recalls, first_f1s, second_precisions, second_recalls, second_f1s):
    max_indices = np.argmax(np.concatenate((np.expand_dims(first_f1s, axis=1),
                                            np.expand_dims(second_f1s, axis=1)), axis=1), axis=1)

    precisions = np.concatenate((np.expand_dims(first_precisions, axis=1),
                                 np.expand_dims(second_precisions, axis=1)), axis=1)
    precisions = np.choose(max_indices, precisions.T)

    recalls = np.concatenate((np.expand_dims(first_recalls, axis=1),
                              np.expand_dims(second_recalls, axis=1)), axis=1)
    recalls = np.choose(max_indices, recalls.T)

    f1s = np.maximum(first_f1s, second_f1s)

    return precisions, recalls, f1s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", default=None, type=str, required=True)
    parser.add_argument("--gold_file", default=None, type=str, required=True)
    parser.add_argument("--relations_file", default=None, type=str, required=True)
    parser.add_argument("--eval_annotated_file", default=None, type=str, required=True)
    parser.add_argument("--test", action='store_true')

    args = parser.parse_args()

    preds = open(args.pred_file, "r", encoding="utf-8-sig").read().splitlines()
    golds = open(args.gold_file, "r", encoding="utf-8-sig").read().splitlines()
    relations = open(args.relations_file, "r", encoding="utf-8-sig").read().splitlines()
    eval_annotations = open(args.eval_annotated_file, "w", encoding="utf-8-sig")

    assert len(preds) == len(golds)

    stance_correct_count = 0
    structurally_correct_graphs_count = 0
    structurally_correct_gold_graphs, structurally_correct_second_gold_graphs, structurally_correct_pred_graphs = [], [], []
    overall_ged = 0.
    for (pred, gold) in zip(preds, golds):
        parts = pred.split("\t")

        assert len(parts) == 2

        pred_stance = parts[0]
        pred_graph = parts[1].lower()

        assert pred_stance in ["support", "counter"]

        parts = gold.split("\t")
        belief = parts[0].lower()
        argument = parts[1].lower()
        gold_stance = parts[2]
        gold_graph = parts[3].lower()
        if args.test:
            second_gold_graph = parts[4].lower()

        # Check for Stance Correctness first
        if pred_stance == gold_stance:
            stance_correct_count += 1
            edges = pred_graph[1:-1].split(")(")
            # Check for Structural Correctness of graphs
            if is_edge_count_correct(edges) and is_graph(edges) and is_edge_structure_correct(edges,
                                                                                              relations) and two_concepts_belief_argument(
                    edges, belief, argument) and is_connected_DAG(edges):
                structurally_correct_graphs_count += 1
                eval_annotations.write(belief + "\t" + pred_graph + "\t" + gold_stance + "\tstruct_correct\n")

                # Save the graphs for Graph Matching or Semantic Correctness Evaluation
                structurally_correct_gold_graphs.append(gold_graph)
                if args.test:
                    structurally_correct_second_gold_graphs.append(second_gold_graph)

                structurally_correct_pred_graphs.append(pred_graph)

                # Compute GED
                ged = get_ged(gold_graph, pred_graph)
                if args.test:
                    ged = min(ged, get_ged(second_gold_graph, pred_graph))
            else:
                eval_annotations.write(belief + "\t" + pred_graph + "\t" + gold_stance + "\tstruct_incorrect\n")
                # GED needs to be computed as the upper bound for structurally incorrect graphs
                ged = get_ged(gold_graph)
                if args.test:
                    ged = min(ged, get_ged(second_gold_graph))
        else:
            # GED also needs to be computed as the upper bound for samples with incorrect stance
            ged = get_ged(gold_graph)
            if args.test:
                ged = min(ged, get_ged(second_gold_graph))
            eval_annotations.write(belief + "\t" + pred_graph + "\t" + gold_stance + "\tstance_incorrect\n")

        overall_ged += ged


    # Evaluate for Graph Matching
    gold_edges = split_to_edges(structurally_correct_gold_graphs)
    second_gold_edges = split_to_edges(structurally_correct_second_gold_graphs) if args.test else None
    pred_edges = split_to_edges(structurally_correct_pred_graphs)

    gold_tokens, pred_tokens, second_gold_tokens = get_tokens(gold_edges, pred_edges, second_gold_edges)

    precisions_rouge, recalls_rouge, f1s_rouge, precisions_bleu, recalls_bleu, f1s_bleu = get_bleu_rouge(
        gold_tokens, pred_tokens, gold_edges, pred_edges)

    precisions_BS, recalls_BS, f1s_BS = get_bert_score(gold_edges, pred_edges)

    # Get max of two gold graphs
    if args.test:
        second_precisions_rouge, second_recalls_rouge, second_f1s_rouge, second_precisions_bleu, second_recalls_bleu, \
        second_f1s_bleu = get_bleu_rouge(second_gold_tokens, pred_tokens, second_gold_edges, pred_edges)

        second_precisions_BS, second_recalls_BS, second_f1s_BS = get_bert_score(second_gold_edges, pred_edges)

        precisions_bleu, recalls_bleu, f1s_bleu = get_max(precisions_bleu, recalls_bleu, f1s_bleu,
                                                          second_precisions_bleu, second_recalls_bleu, second_f1s_bleu)
        precisions_rouge, recalls_rouge, f1s_rouge = get_max(precisions_rouge, recalls_rouge, f1s_rouge,
                                                             second_precisions_rouge, second_recalls_rouge,
                                                             second_f1s_rouge)
        precisions_BS, recalls_BS, f1s_BS = get_max(precisions_BS, recalls_BS, f1s_BS,
                                                    second_precisions_BS, second_recalls_BS, second_f1s_BS)


    print(f'Stance Accuracy (SA): {stance_correct_count / len(golds):.4f}')
    print(f'Structural Correctness Accuracy (StCA): {structurally_correct_graphs_count / len(golds):.4f}')

    print(f'G-BLEU Precision: {precisions_bleu.sum() / len(golds):.4f}')
    print(f'G-BLEU Recall: {recalls_bleu.sum() / len(golds):.4f}')
    print(f'G-BLEU F1: {f1s_bleu.sum() / len(golds):.4f}\n')

    print(f'G-Rouge Precision: {precisions_rouge.sum() / len(golds):.4f}')
    print(f'G-Rouge Recall Score: {recalls_rouge.sum() / len(golds):.4f}')
    print(f'G-Rouge F1 Score: {f1s_rouge.sum() / len(golds):.4f}')

    print(f'G-BertScore Precision Score: {precisions_BS.sum() / len(golds):.4f}')
    print(f'G-BertScore Recall Score: {recalls_BS.sum() / len(golds):.4f}')
    print(f'G-BertScore F1 Score: {f1s_BS.sum() / len(golds):.4f}\n')

    print(f'Graph Edit Distance (GED): {overall_ged / len(golds):.4f}\n')
