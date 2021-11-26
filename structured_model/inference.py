from pulp import *
import numpy as np

def merge_nodes(no_edge_prob, max_edge_prob, max_edge_index, ordered_nodes):
    new_ordered_nodes = []
    indices_to_remove = []
    for (i, ordered_node) in enumerate(ordered_nodes):
        if ordered_node not in new_ordered_nodes:
            new_ordered_nodes.append(ordered_node)
        else:
            indices_to_remove.append(i)

    assert len(new_ordered_nodes) <= 8

    if len(indices_to_remove) == 0:
        return no_edge_prob, max_edge_prob, max_edge_index, new_ordered_nodes

    new_no_edge_prob = np.delete(no_edge_prob, indices_to_remove, axis=0)
    new_no_edge_prob = np.delete(new_no_edge_prob, indices_to_remove, axis=1)

    new_max_edge_prob = np.delete(max_edge_prob, indices_to_remove, axis=0)
    new_max_edge_prob = np.delete(new_max_edge_prob, indices_to_remove, axis=1)

    new_max_edge_index = np.delete(max_edge_index, indices_to_remove, axis=0)
    new_max_edge_index = np.delete(new_max_edge_index, indices_to_remove, axis=1)

    return new_no_edge_prob, new_max_edge_prob, new_max_edge_index, new_ordered_nodes

def solve_LP_no_connectivity(no_edge_prob, max_edge_prob, max_edge_index, ordered_nodes, edge_label_list):
    no_edge_prob, max_edge_prob, max_edge_index, ordered_nodes = merge_nodes(no_edge_prob, max_edge_prob,
                                                                             max_edge_index, ordered_nodes)
    prob = LpProblem("Node edge consistency ", LpMaximize)
    all_vars = {}

    # Optimization Problem
    opt_prob = None
    for i in range(len(no_edge_prob)):
        for j in range(len(no_edge_prob)):
            var0 = LpVariable("Edge_" + str(i + 1) + "_" + str(j + 1) + "_0", 0, 1, LpInteger)
            var1 = LpVariable("Edge_" + str(i + 1) + "_" + str(j + 1) + "_1", 0, 1, LpInteger)

            all_vars[(i, j, 0)] = var0
            all_vars[(i, j, 1)] = var1

            if opt_prob is None:
                opt_prob = no_edge_prob[i][j] * all_vars[(i, j, 0)] + max_edge_prob[i][j] * all_vars[(i, j, 1)]
            else:
                opt_prob += no_edge_prob[i][j] * all_vars[(i, j, 0)] + max_edge_prob[i][j] * all_vars[(i, j, 1)]

    prob += opt_prob, "Maximum Score"

    # An edge is either present or absent
    for i in range(len(no_edge_prob)):
        for j in range(len(no_edge_prob)):
            prob += all_vars[(i, j, 0)] + all_vars[(i, j, 1)] == 1, "Exist condition" + str(i) + "_" + str(j)

    prob.solve()

    edges = []
    edges_dict = {}
    for v in prob.variables():
        if v.varValue > 0 and v.name.endswith("1") and v.name.startswith("Edge"):
            name = v.name.split("_")
            n_i = int(name[1]) - 1
            n_j = int(name[2]) - 1
            assert ordered_nodes[n_i] != ordered_nodes[n_j]
            assert (ordered_nodes[n_i], ordered_nodes[n_j]) not in edges_dict
            edge = "(" + ordered_nodes[n_i] + "; " + edge_label_list[max_edge_index[n_i][n_j]] + "; " + ordered_nodes[
                n_j] + ")"
            edges.append(edge)
            edges_dict[(ordered_nodes[n_i], ordered_nodes[n_j])] = True
    print("Max score = ", value(prob.objective))

    return edges


def solve_LP(no_edge_prob, max_edge_prob, max_edge_index, ordered_nodes, edge_label_list):
    no_edge_prob, max_edge_prob, max_edge_index, ordered_nodes = merge_nodes(no_edge_prob, max_edge_prob, max_edge_index, ordered_nodes)
    prob = LpProblem("Node edge consistency ", LpMaximize)
    all_vars = {}

    all_flow_vars = {}

    source_id = -1
    sink_id = -2

    print(ordered_nodes)

    node_ids_present = [i for i in range(len(ordered_nodes))]

    # add flow from source to one node present
    # arbitarily choosing that node to be the last node
    # 1000 is infinity
    all_flow_vars[(source_id, node_ids_present[-1])] = \
        LpVariable("Flow_source_" + str(node_ids_present[-1] + 1), 0, 1000, LpInteger)

    # add flow from all nodes present to sink
    for i in range(len(node_ids_present)):
        temp = node_ids_present[i]
        all_flow_vars[(temp, sink_id)] = LpVariable("Flow_" + str(temp + 1) + "_sink", 0, 1000, LpInteger)

    # define capacities
    C = {}
    # capacity from source to 1st node is number of nodes in graph
    C[(source_id, node_ids_present[-1])] = len(node_ids_present)
    C[(node_ids_present[-1], source_id)] = 0

    # capacity from nodes in graph to sink is 1
    for i in range(len(node_ids_present)):
        temp = node_ids_present[i]
        C[(temp, sink_id)] = 1
        C[(sink_id, temp)] = 0

    # capacities inside graph are infinite or say 1000 in this case, except self loops and if the edge is not possible
    arcs = set()
    for i in range(len(no_edge_prob)):
        for j in range(len(no_edge_prob)):
            if (i == j) or (i not in node_ids_present) or (j not in node_ids_present):
                C[(i, j)] = 0
            else:
                C[(i, j)] = 1000
                arcs.add((i, j))
                arcs.add((j, i))
    arcs = list(arcs)

    # Optimization Problem
    opt_prob = None
    for i in range(len(no_edge_prob)):
        for j in range(len(no_edge_prob)):
            if i == j:
                continue
            var0 = LpVariable("Edge_" + str(i + 1) + "_" + str(j + 1) + "_0", 0, 1, LpInteger)
            var1 = LpVariable("Edge_" + str(i + 1) + "_" + str(j + 1) + "_1", 0, 1, LpInteger)

            all_vars[(i, j, 0)] = var0
            all_vars[(i, j, 1)] = var1

            f_var = LpVariable("Flow_" + str(i + 1) + "_" + str(j + 1), 0, 1000, LpInteger)
            all_flow_vars[(i, j)] = f_var

            if opt_prob is None:
                opt_prob = no_edge_prob[i][j] * all_vars[(i, j, 0)] + max_edge_prob[i][j] * all_vars[(i, j, 1)]
            else:
                opt_prob += no_edge_prob[i][j] * all_vars[(i, j, 0)] + max_edge_prob[i][j] * all_vars[(i, j, 1)]

    prob += opt_prob, "Maximum Score"

    # Constraints
    for i in range(len(no_edge_prob)):
        for j in range(len(no_edge_prob)):
            if i == j:
                continue
            # An edge can either be present or not present
            prob += all_vars[(i, j, 0)] + all_vars[(i, j, 1)] == 1, "Exist condition" + str(i) + "_" + str(j)

            # flow less than capacity
            prob += all_flow_vars[(i, j)] <= C[(i, j)], "Capacity constraint " + str(i) + " " + str(j)

    # capacity constraint of source to 1st node
    prob += all_flow_vars[(source_id, node_ids_present[-1])] <= C[
        (source_id, node_ids_present[-1])], "Capacity constraint source " + str(node_ids_present[-1])

    # capacity constraint of nodes to sink
    for i in range(len(node_ids_present)):
        temp = node_ids_present[i]
        prob += all_flow_vars[(temp, sink_id)] == C[(temp, sink_id)], "Capacity constraint " + str(temp) + " sink"

    # node flow conservation constraint
    for n in range(len(no_edge_prob)):
        if n == node_ids_present[-1]:
            prob += (all_flow_vars[(source_id, n)] + lpSum([all_flow_vars[(i, j)] for (i, j) in arcs if j == n]) ==
                     lpSum([all_flow_vars[(i, j)] for (i, j) in arcs if i == n]) + all_flow_vars[(n, sink_id)]), \
                    "Flow Conservation in Node " + str(n)
        else:
            prob += (lpSum([all_flow_vars[(i, j)] for (i, j) in arcs if j == n]) ==
                     lpSum([all_flow_vars[(i, j)] for (i, j) in arcs if i == n]) + all_flow_vars[(n, sink_id)]), \
                    "Flow Conservation in Node " + str(n)

    # Max flow should be equal to number of nodes in graph
    # to ensure this make the flow from source exactly equal to capacity
    # also ensure that the flow occurs only when the edge exists
    prob += all_flow_vars[(source_id, node_ids_present[-1])] == C[(source_id, node_ids_present[-1])]
    for i in range(len(no_edge_prob)):
        for j in range(len(no_edge_prob)):
            if i == j:
                continue
            prob += len(node_ids_present) * (all_vars[i, j, 1] + all_vars[j, i, 1]) - all_flow_vars[
                (i, j)] >= 0, "Valid flow " + str(
                i + 1) + " " + str(j + 1)

    prob.solve()

    edges = []
    edges_dict = {}
    for v in prob.variables():
        if v.varValue > 0 and v.name.endswith("1") and v.name.startswith("Edge"):
            name = v.name.split("_")
            if name[1] == 'source' or name[2] == 'sink' or name[2] == 'source' or name[1] == 'sink':
                continue
            n_i = int(name[1]) - 1
            n_j = int(name[2]) - 1
            assert ordered_nodes[n_i] != ordered_nodes[n_j]
            assert (ordered_nodes[n_i], ordered_nodes[n_j]) not in edges_dict
            edge = "(" + ordered_nodes[n_i] + "; " + edge_label_list[max_edge_index[n_i][n_j]] + "; " + ordered_nodes[n_j] + ")"
            edges.append(edge)
            edges_dict[(ordered_nodes[n_i], ordered_nodes[n_j])] = True

    print("Max score = ", value(prob.objective))

    print(edges)
    return edges
