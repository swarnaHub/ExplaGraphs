## Commonsense-augmented Structured Prediction Model

The structured model for graph generation has multiple modules. Follow the below steps in order to train and test the model. Note that all these scripts should be executed from the root folder.

# Step 1: 

We'll first compute the embeddings of the relations using a pre-trained LM (RoBERTa) and save them inside ```data``` folder. Alternatively, you can find these embeddings in ```data/relations.pt```.

```
python structured_model/save_relation_embeddings.py
```

# Step 2:

In order to leverage commonsense knowledge from ConceptNet, next we'll fine-tune RoBERTa on ConceptNet. You can do so using the below script and the training data can be found [here](https://drive.google.com/drive/folders/19faqrwXLM5EySeB4yQ3JRPzGsi68DF07?usp=sharing). Alternatively, directly download our pre-trained model [here](https://drive.google.com/drive/folders/14CnyJUQX8Z2rubwofDGvTLnh_3bLsjml?usp=sharing).

```
bash model_scripts/train_conceptnet_finetuning.sh
```

# Step 3:

Next, we'll use the previously finetuned model to train our graph generation model using the below script. A couple of things to keep in mind: (1) This model has a component which predicts the external nodes first, which we obtain using BART. These are comma separated concepts as uploaded in ```data/external_concepts_dev.txt```. (2) Once you have trained the model, it will save the internal node predictions as uploaded in ```data/internal_concepts_dev.txt```. These are in BIO format where each stretch of B-N to I-N denotes a node.

```
bash model_scripts/train_structured_model.sh
```

# Step 4:

You can directly download our trained model [here](https://drive.google.com/drive/folders/1fD0BqkigLdxXfR_tLrMnB7CTewsGx_HL?usp=sharing) and test it to generate the final graphs. Note that graph generation uses three predictions -- (1) internal node predictions, (2) external node predictions and (3) the edge logits. All these come together in an ILP to generate the final graphs. The below script handles all of these and will generate the graphs in ```prediction_edges_dev.lst``` inside the model folder. 

Once you download our pre-trained model, you'll also find our generated graphs, so you can use them to directly obtain the metrics.

```
bash model_scripts/test_structured_model.sh
```
