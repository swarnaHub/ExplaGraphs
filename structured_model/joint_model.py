from pytorch_transformers import BertPreTrainedModel, RobertaConfig, \
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaModel
from pytorch_transformers.modeling_roberta import RobertaClassificationHead
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn


class NodeClassificationHead(nn.Module):
    def __init__(self, config, num_labels_node):
        super(NodeClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels_node)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


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


class RobertaForEX(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForEX, self).__init__(config)

        self.num_labels_node = 3 # 3-way classification for B-N, I-N, O
        self.num_labels_edge = 29  # 29-way classification for 28 relations and 1 no edge
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_node = NodeClassificationHead(config, self.num_labels_node)
        self.classifier_edge = EdgeClassificationHead(config)

        self.relation_embeddings = torch.load("./data/relations.pt").to("cuda")

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, node_start_index=None, node_end_index=None,
                node_label=None,
                edge_label=None, stance_label=None, position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, head_mask=head_mask)

        loss_fct = CrossEntropyLoss()
        sequence_output = outputs[0]

        # Node sequence tagging loss
        node_logits = self.classifier_node(self.dropout(sequence_output))
        node_loss = loss_fct(node_logits.view(-1, self.num_labels_node), node_label.view(-1))

        # Edge embedding computation
        max_edges = edge_label.shape[1]
        batch_size = node_label.shape[0]
        embedding_dim = sequence_output.shape[2]

        batch_edge_embedding = torch.zeros((batch_size, max_edges, self.relation_embeddings.shape[0], 4 * embedding_dim)).to("cuda")

        for batch_index in range(batch_size):
            sample_node_embedding = None
            count = 0
            for (start_index, end_index) in zip(node_start_index[batch_index], node_end_index[batch_index]):
                if start_index == 0:
                    break
                else:
                    node_embedding = torch.mean(sequence_output[batch_index, start_index:(end_index+1), :],
                                                dim=0).unsqueeze(0)
                    count += 1
                    if sample_node_embedding is None:
                        sample_node_embedding = node_embedding
                    else:
                        sample_node_embedding = torch.cat((sample_node_embedding, node_embedding), dim=0)

            repeat1 = sample_node_embedding.unsqueeze(0).repeat(len(sample_node_embedding), 1, 1)
            repeat2 = sample_node_embedding.unsqueeze(1).repeat(1, len(sample_node_embedding), 1)
            sample_edge_embedding = torch.cat((repeat1, repeat2, (repeat1 - repeat2)), dim=2)

            sample_edge_embedding = sample_edge_embedding.view(-1, sample_edge_embedding.shape[-1])

            relation_embedding = self.relation_embeddings.unsqueeze(0).repeat(sample_edge_embedding.shape[0], 1, 1)
            sample_edge_embedding = sample_edge_embedding.unsqueeze(1).repeat(1, relation_embedding.shape[1], 1)

            sample_edge_embedding_with_relation = torch.cat((sample_edge_embedding, relation_embedding), dim=2)

            # Append 0s at the end (these will be ignored for loss)
            sample_edge_embedding_with_relation = torch.cat((sample_edge_embedding_with_relation,
                                               torch.zeros(
                                                   (max_edges - len(sample_edge_embedding), relation_embedding.shape[1]
                                                    , 4 * embedding_dim)).to("cuda")), dim=0)

            batch_edge_embedding[batch_index, :, :, :] = sample_edge_embedding_with_relation

        # Edge loss
        edge_logits = self.classifier_edge(batch_edge_embedding.view(batch_size, max_edges, -1))
        edge_loss = loss_fct(edge_logits.view(-1, self.num_labels_edge), edge_label.view(-1))
        total_loss = node_loss + edge_loss

        outputs = (node_logits, edge_logits) + outputs[2:]
        outputs = (total_loss, node_loss, edge_loss) + outputs

        return outputs  # (total_loss), node_loss, edge_loss, node_logits, edge_logits,
        # (hidden_states), (attentions)
