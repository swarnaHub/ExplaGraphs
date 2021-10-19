# ExplaGraphs
Dataset and PyTorch code for our EMNLP 2021 paper:

[ExplaGraphs: An Explanation Graph Generation Task for Structured Commonsense Reasoning](https://arxiv.org/abs/2104.07644)

[Swarnadeep Saha](https://swarnahub.github.io/), [Prateek Yadav](https://prateek-yadav.github.io/), [Lisa Bauer](https://www.cs.unc.edu/~lbauer6/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

## Website and Leaderboard
ExplaGraphs is hosted [here](https://explagraphs.github.io/).
You can find the leaderboard, a brief discussion of our dataset, evaluation metrics and some notes about how to submit predictions on the test set.

## Installation
This repository is tested on Python 3.8.3.  
You should install ExplaGraphs on a virtual environment. All dependencies can be installed as follows:
```
pip install -r requirements.txt
```

## Dataset
ExplaGraphs dataset can be found inside the ```data``` folder.

It contains the training data in ```train.tsv``` and the validation samples in ```dev.tsv```.

Each training sample contains four tab-separated entries -- belief, argument, stance label and the explanation graph.

The graph is organized as a bracketed string ```(edge_1)(edge_2)...(edge_n)```, where each edge is of the form ```concept_1; relation; concept_2```. 

## Evaluation Metrics
ExplaGraphs is a joint task that requires predicting both the stance label and the corresonding explanation graph. Independent of how you choose to represent the graphs in your models, you must represent the graphs as bracketed strings (as in our training data) in order to use our evaluation scripts.

We propose multiple evaluation metrics as detailed in Section 6 of our paper. Below we provide the steps to use our evaluation scripts.

### Step 1
First, we evaluate the graphs against all the non-model based metrics. This includes computing the stance accuracy (SA), Structural Correctness Accuracy for Graphs (StCA), G-BertScore (G-BS) and Graph Edit Distance (GED). Run the following script to get these.
```
bash eval_scripts/eval_first
```
This takes as input the gold file, predictions file, and the relations file and outputs an intermediate file ```annotations.tsv```. In this intermediate file, each sample is annotated with one of the three labels -- ```stance_incorrect```, ```struct_incorrect``` and ```struct_correct```. The first label denotes the samples where the predicted stance is incorrect, the second denotes the ones where the stance is correct but the graph is structurally incorrect and the third denotes the ones where the stance is correct and the graph is also structurally correct.

Structural Correctness Evaluation requires satisfying all the constraints we define for the task, which include the graph be connected DAG with at least three edges and having at least two exactly matching concepts from the belief and two from the argument. You **SHOULD NOT** look to boost this accuracy up by some arbitrary post-hoc correction of structurally incorrect graphs (like adding a random edge to make a disconnected graph connected). 

Note that our evaluation framework is a pipeline, so the G-BS and GED metrics are computed only on the fraction of samples with annotation ```struct_correct```.

### Step 2
Given this intermediate annotation file, we'll now compute the Semantic Correctness Accuracy for Graphs (SeCA). Once again, this will only evaluate the fraction of samples where the stance is correct and the graphs are structurally correct. It is a model-based metric and we release our pre-trained model [here](). Once you download the model, run the following script to get SeCA.
```
bash eval_scripts/eval_seca
```

### Step 3
In the final step, we compute the Edge Importance Accuracy (EA). This is again a model-based metric and you can download the pre-trained model [here](). Once you download the model, run the following script
```
bash eval_scripts/eval_ea
```
This measures the importance of an edge by removing it from the predicted graph and checking for the difference in stance confidence (with and without it) according to the model. An increase denotes that the edge is important while a decrease suggests otherwise.

## Evaluating on the Test Set

To evaluate your model on the test set, please email us at swarna@cs.unc.edu with your model (or a link to download it), a detailed README with the installation environment (requirements.txt) and a script to generate the predictions.

The predictions should be generated in a ```tsv``` file with each line containing two tab-separated entries, first the predicted stance (support/counter) followed by the predicted graph in the same bracketed format as in the train and validation files. A sample prediction file in shown inside ```data``` folder.

For all latest results on ExplaGraphs, please refer to the leaderboard [here](https://explagraphs.github.io/).

## Baseline Models
We are in the process of releasing our baseline models. Stay tuned!

### Citation
```
@inproceedings{saha2021explagraphs,
  title={ExplaGraphs: An Explanation Graph Generation Task for Structured Commonsense Reasoning},
  author={Saha, Swarnadeep and Yadav, Prateek and Bauer, Lisa and Bansal, Mohit},
  booktitle={EMNLP},
  year={2021}
}
```
