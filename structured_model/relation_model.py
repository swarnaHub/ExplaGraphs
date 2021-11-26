from pytorch_transformers import BertPreTrainedModel, RobertaConfig, \
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaModel
from pytorch_transformers.modeling_roberta import RobertaClassificationHead
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn

class EdgeClassificationHead(nn.Module):
    def __init__(self, config):
        super(EdgeClassificationHead, self).__init__()
        self.dense = nn.Linear(29 * 4 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 29)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForRelationPrediction(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForRelationPrediction, self).__init__(config)

        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier_edge = EdgeClassificationHead(config)

        self.relation_embeddings = torch.load("./data/relations.pt")

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_indices=None, end_indices=None, relation_label=None,
                position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, head_mask=head_mask)

        loss_fct = CrossEntropyLoss()
        sequence_output = outputs[0]
        batch_size = relation_label.shape[0]
        embedding_dim = sequence_output.shape[2]
        self.relation_embeddings = self.relation_embeddings.to(sequence_output)
        batch_edge_embedding = torch.zeros((batch_size, self.relation_embeddings.shape[0], 4 * embedding_dim)).to(sequence_output)

        for batch_index in range(batch_size):
            concept1_start_index = start_indices[batch_index][0]
            concept1_end_index = end_indices[batch_index][0]
            concept1_embedding = torch.mean(sequence_output[batch_index, concept1_start_index:(concept1_end_index+1), :]
                                            , dim=0).unsqueeze(0)

            concept2_start_index = start_indices[batch_index][1]
            concept2_end_index = end_indices[batch_index][1]
            concept2_embedding = torch.mean(sequence_output[batch_index, concept2_start_index:(concept2_end_index+1), :]
                                            , dim=0).unsqueeze(0)

            edge_embedding = torch.cat((concept1_embedding, concept2_embedding,
                                        (concept1_embedding - concept2_embedding)), dim=1)
            edge_embedding_with_relation = torch.cat((edge_embedding.repeat(self.relation_embeddings.shape[0], 1),
                                                      self.relation_embeddings), dim=1)
            batch_edge_embedding[batch_index, :, :] = edge_embedding_with_relation.unsqueeze(0)

        logits = self.classifier_edge(batch_edge_embedding.view(batch_size, -1))
        loss = loss_fct(logits.view(-1, self.num_labels), relation_label.view(-1))

        outputs = (loss, logits) + outputs

        return outputs
