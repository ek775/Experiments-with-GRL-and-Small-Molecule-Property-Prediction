# import torch components
import torch
import torch.nn.functional as F
import torch_geometric.nn as PyG

# Encoder
class GACsol(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=64, embed_channels=32, aggr=PyG.aggr.SoftmaxAggregation(learn=True)):
        super().__init__()
        # message passing layers
        self.conv1 = PyG.GATv2Conv(in_channels, 
                                 hidden_dim, 
                                 heads=16, 
                                 dropout=0.1,
                                 add_self_loops=False, 
                                 aggr=aggr) 
        self.lin1 = PyG.Linear(-1, hidden_dim)
        self.conv2 = PyG.GATConv(hidden_dim, 
                                 embed_channels, 
                                 heads=16,
                                 dropout=0.1, 
                                 add_self_loops=False, 
                                 aggr=aggr)
        self.lin2 = PyG.Linear(-1, embed_channels)
        # pooling layer
        self.pool3 = PyG.pool.SAGPooling(embed_channels)
        # regression out
        self.out = PyG.Linear(-1, 1)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        # message passing
        x = self.conv1(x, edge_index) #+ self.lin1(x)
        x = x.relu()
        #x = self.conv2(x, edge_index) #+ self.lin2(x)
        x = x.relu()
        # pooling
        x = self.pool3(x, edge_index, edge_attr).relu()
        return self.out(x) 
