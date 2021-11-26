from transformers import RobertaTokenizer, RobertaModel
import torch

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaModel.from_pretrained('roberta-large')
    relations = open("./data/relations.txt", "r").read().splitlines()
    relations.append("no relation")
    embeddings = None
    for relation in relations:
        inputs = tokenizer(relation, return_tensors="pt")
        embedding = model(**inputs)[1]
        if embeddings is None:
            embeddings = embedding
        else:
            embeddings = torch.cat((embeddings, embedding), dim=0)

    torch.save(embeddings, "./data/relations.pt")
