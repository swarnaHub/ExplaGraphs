import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_probs", default=None, type=str, required=True)
    parser.add_argument("--final_probs", default=None, type=str, required=True)
    parser.add_argument("--initial_file", default=None, type=str, required=True)
    parser.add_argument("--final_file", default=None, type=str, required=True)
    parser.add_argument("--gold_file", default=None, type=str, required=True)

    args = parser.parse_args()

    golds = open(args.gold_file, "r", encoding="utf-8-sig").read().splitlines()
    initial_probs = open(args.initial_probs, "r", encoding="utf-8-sig").read().splitlines()[1:]
    final_probs = open(args.final_probs, "r", encoding="utf-8-sig").read().splitlines()[1:]
    initial_samples = open(args.initial_file, "r", encoding="utf-8-sig").read().splitlines()
    final_samples = open(args.final_file, "r", encoding="utf-8-sig").read().splitlines()

    index_to_line = {}
    gold_labels = {}
    for (i, sample) in enumerate(final_samples):
        index = int(sample.split("\t")[0])
        if index not in index_to_line:
            index_to_line[index] = [i]
        else:
            index_to_line[index].append(i)
        gold_labels[index] = sample.split("\t")[3]

    initial_probs_converted = {}
    for (initial_sample, initial_prob) in zip(initial_samples, initial_probs):
        temp = [float(element) for element in initial_prob[1:-1].split(" ") if element != ""]
        assert len(temp) == 2
        index = initial_sample.split("\t")[0]
        initial_probs_converted[int(index)] = temp

    new_probs_converted = []
    for final_prob in final_probs:
        temp = [float(element) for element in final_prob[1:-1].split(" ") if element != ""]
        assert len(temp) == 2
        new_probs_converted.append(temp)

    macro_increment_count = 0
    label_to_index = {"support": 0, "counter": 1}
    score_list = []
    for index in index_to_line:
        lines = index_to_line[index]
        label = gold_labels[index]

        sample_increment_count = 0
        for line in lines:
            temp_final = new_probs_converted[line][label_to_index[label]]
            temp_initial = initial_probs_converted[index][label_to_index[label]]
            # An edge is important if it causes an increase in stance confidence
            if temp_initial > temp_final:
                sample_increment_count += 1

        # Average across all edges
        macro_increment_count += sample_increment_count / len(lines)

    # Average across all samples
    print(f'Edge Importance Accuracy (EA): {macro_increment_count / len(golds):.4f}')

