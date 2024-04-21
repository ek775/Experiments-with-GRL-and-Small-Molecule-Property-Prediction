# import torch components
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import GAE
from torch_geometric.nn.models import GCN
import torch_geometric.nn as PyG

# Encoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, aggr=PyG.aggr.SoftmaxAggregation(learn=True)):
        super(GCNEncoder, self).__init__()
        self.conv1 = PyG.GATConv(in_channels, 
                                 64, 
                                 heads=8, 
                                 dropout=0.1,
                                 add_self_loops=False, 
                                 aggr=aggr) 
        self.conv2 = PyG.GATConv(-1, 
                                 out_channels, 
                                 heads=8,
                                 dropout=0.1, 
                                 add_self_loops=False, 
                                 aggr=aggr)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index).relu()