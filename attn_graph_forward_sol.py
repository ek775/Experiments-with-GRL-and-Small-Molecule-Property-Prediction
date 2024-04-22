# import torch components
import torch
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.transforms import SVDFeatureReduction
# Model
class GACsol(torch.nn.Module):
    def __init__(self, hidden_dim=64, embed_channels=32, aggr=PyG.aggr.SoftmaxAggregation(learn=True)):
        super().__init__()
        # message passing layers
        self.conv1 = PyG.GATv2Conv((-1,-1), 
                                 hidden_dim, 
                                 dropout=0.1,
                                 add_self_loops=False,
                                 aggr=aggr) 
        self.lin1 = PyG.Linear(-1, hidden_dim) # skip connection
        self.conv2 = PyG.GATv2Conv((-1,-1), 
                                 embed_channels,
                                 dropout=0.1, 
                                 add_self_loops=False,
                                 aggr=aggr)
        self.lin2 = PyG.Linear(-1, embed_channels) # skip connection
        # MLP and regression out
        self.lin3 = PyG.Linear(-1, embed_channels)
        #self.lin4 = PyG.Linear(-1, 1)
        self.transform = SVDFeatureReduction(8)
        self.out = torch.nn.Linear(8,1)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        # message passing
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index) + self.lin2(x)
        x = F.leaky_relu(x)
        # MLP
        x = self.lin3(x)
        x = F.relu(x)
        x = self.transform(x)
        return self.out(x)
