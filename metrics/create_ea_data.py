import argparse
import networkx as nx


def get_dfs_ordering(graph_string):
    graph = nx.DiGraph()
    nodes = []
    relations = {}
    for edge in graph_string[1:-1].split(")("):
        parts = edge.split("; ")
        graph.add_edge(parts[0], parts[2])
        if parts[0] not in nodes:
            nodes.append(parts[0])
        if parts[2] not in nodes:
            nodes.append(parts[2])
        relations[(parts[0], parts[2])] = parts[1]

    in_degrees = list(graph.in_degree(nodes))

    start_nodes = []
    for (i, node) in enumerate(nodes):
        if in_degrees[i][1] == 0:
            start_nodes.append(in_degrees[i][0])

    dfs_edges = list(nx.edge_dfs(graph, source=start_nodes))

    new_graph_string = ""
    for edge in dfs_edges:
        new_graph_string += "(" + edge[0] + "; " + relations[(edge[0], edge[1])] + "; " + edge[1] + ")"

    return new_graph_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", default=None, type=str, required=True)
    parser.add_argument("--eval_annotated_file", default=None, type=str, required=True)
    parser.add_argument("--output_EA_initial", default=None, type=str, required=True)
    parser.add_argument("--output_EA_final", default=None, type=str, required=True)

    args = parser.parse_args()
    golds = open(args.gold_file, "r", encoding="utf-8-sig").read().splitlines()
    preds = open(args.eval_annotated_file, "r", encoding="utf-8-sig").read().splitlines()

    assert len(golds) == len(preds)

    output_initial = open(args.output_EA_initial, "w", encoding="utf-8-sig")
    output_final = open(args.output_EA_final, "w", encoding="utf-8-sig")

    for i, (gold, pred) in enumerate(zip(golds, preds)):
        gold_parts = gold.split("\t")
        belief, argument, stance = gold_parts[0], gold_parts[1], gold_parts[2]

        pred_parts = pred.split("\t")
        if pred_parts[3] != "struct_correct":
            continue

        graph = get_dfs_ordering(pred_parts[1])
        for edge in graph[1:-1].split(")("):
            leave_one_out_graph = graph.replace("(" + edge + ")", "")
            leave_one_out_graph = leave_one_out_graph.replace("(", "").replace(";", "").replace(")", ". ")
            leave_one_out_argument = argument + " " + leave_one_out_graph
            output_final.write(str(i) + "\t" + belief + "\t" + leave_one_out_argument + "\t" + stance + "\n")

        whole_graph_argument = argument + " " + graph.replace("(", "").replace(";", "").replace(")", ". ")
        output_initial.write(str(i) + "\t" + belief + "\t" + whole_graph_argument + "\t" + stance + "\n")
