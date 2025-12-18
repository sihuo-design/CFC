import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.data import Data

# from gnn_node import GNN_node_TU

from torch_geometric.nn import (GCNConv, GINConv, GATConv, MLP)

from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_sparse
from torch_sparse import SparseTensor, matmul
import scipy.sparse
import numpy as np


class GCNNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCNNet, self).__init__()

        self.args = args
        self.conv1 = GCNConv(dataset.num_features,
                             args.hidden_dim1,
                             cached= True,
                             add_self_loops = True,
                             normalize = True)
        self.conv2 = GCNConv(args.hidden_dim1,
                            args.hidden_dim2,
                            cached= True,
                            add_self_loops = True,
                            normalize = True)
        self.clf1 = nn.Linear(args.hidden_dim2, len(args.known_class))
        self.dropout = args.dropout


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, None

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf1(x)
        return x #F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.conv1 = nn.Linear(dataset.num_features, args.hidden_dim1)
        self.conv2 = nn.Linear(args.hidden_dim1, args.hidden_dim2)
        self.clf1 = nn.Linear(args.hidden_dim2, len(args.known_class))
        self.bn1 = nn.BatchNorm1d(args.hidden_dim1)
        self.bn2 = nn.BatchNorm1d(args.hidden_dim2)
    
    def forward(self, g):
        x = g.x
        edge_index = g.edge_index
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.clf1(x)
        
        return x

class GINNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super().__init__()

        in_channels = dataset.num_features
        hidden_channels = args.hidden_dim
        out_channels = dataset.num_classes

        self.convs = torch.nn.ModuleList()
        for _ in range(2):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, None
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.mlp(x)


class GATNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GATNet, self).__init__()

        self.args = args
        self.conv1 = GATConv(dataset.num_features, args.hidden_dim, heads=8)
        self.conv2 = GATConv(args.hidden_dim * 8, dataset.num_classes, heads=1, concat=False, dropout=0.6)


    def forward(self, data):

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x #x.log_softmax(dim=-1)

