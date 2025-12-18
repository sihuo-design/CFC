import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch import optim
import numpy as np

import sys
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from transformers import AutoModel
from typing import List, Union, Tuple

class GCN_model2(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat, num_layers):
        super().__init__()
        self.Conv1 = GCNConv(in_feat, h_feat)
        self.Conv2 = GCNConv(h_feat, out_feat)
        self.clf1 = nn.Linear(out_feat, num_layers)
    
    def forward(self, seq, edge_index):
        res1 = self.Conv1(seq, edge_index)
        res2 = self.Conv2(res1, edge_index)
        res3 = self.clf1(res2)

        return res3

class MLP(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat, num_layers):
        super().__init__()
        self.Conv1 = nn.Linear(in_feat, h_feat)
        self.Conv2 = nn.Linear(h_feat, out_feat)
        self.clf1 = nn.Linear(out_feat, num_layers)
    
    def forward(self, seq, edge_index):
        res1 = self.Conv1(seq)
        res2 = self.Conv2(res1)
        res3 = self.clf1(res2)
        
        return res3

class labelConv(MessagePassing):
    # def __init__(self, in_channels, out_channels):
    #     super(labelConv, self).__init__(aggr='add')
    def label_propagate(self, feature, edge_index, train_indices, selected_unseen_indices, order=2, drop_rate=0.1):
        #feature = F.dropout(feature, args.dropout, training=training)
        x = feature
        edge_index, _ = add_self_loops(edge_index)
        train_mask = torch.zeros(x.shape[0], dtype=torch.bool).cuda()
        train_mask[train_indices] = True

        n = x.shape[0]
        drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
        
        train_label = feature[train_mask,:]
        #y = torch.zeros([x.shape[0],x.shape[1]]).cuda()
        for i in range(order):
            masks = torch.bernoulli(1. - drop_rates).unsqueeze(1).cuda()
            x = masks * x
            # x = torch.spmm(edge_index, x)
            x = self.propagate(edge_index, x=x, edge_weight=None)

            x[train_mask, :] = train_label

        return x.detach()
    

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Load pre-trained language model
        self.language_model = AutoModel.from_pretrained(self.model_name)
        hidden_size = self.language_model.config.hidden_size

        # Freeze the language model's parameters
        # for param in self.language_model.parameters():
        #     param.requires_grad = False

        # Classification layer
        self.classification_model = nn.Linear(
            in_features=hidden_size,
            out_features=self.num_classes,
        )

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor = None,
        attention_mask: Tensor = None,
        return_features: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        transformer_output = self.language_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        # Use pooler_output if available, else fallback to mean of last_hidden_state
        if hasattr(transformer_output, "pooler_output") and transformer_output.pooler_output is not None:
            embeddings = transformer_output.pooler_output
        else:
            embeddings = transformer_output.last_hidden_state.mean(dim=1)

        # Classification output
        output = self.classification_model(embeddings)

        if return_features:
            return output, embeddings

        return output
