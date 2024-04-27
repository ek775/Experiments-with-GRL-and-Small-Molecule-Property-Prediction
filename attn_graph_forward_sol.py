# import torch components
import torch
import torch.nn.functional as F
import torch_geometric.nn as PyG
    
# Model
class GACsol(torch.nn.Module):
    def __init__(self, 
                 hidden_dim=16, 
                 embed_channels=8, 
                 aggr=PyG.aggr.SoftmaxAggregation(learn=True)):
        super().__init__()
        # message passing layers
        self.conv1 = PyG.GATv2Conv((-1,-1), 
                                 hidden_dim, 
                                 dropout=0.1,
                                 add_self_loops=False,
                                 aggr=aggr) 
        self.lin1 = PyG.Linear(-1, hidden_dim) # skip connection
        self.norm1 = PyG.norm.BatchNorm(hidden_dim) # batch normalization

        self.conv2 = PyG.GATv2Conv((-1,-1), 
                                 embed_channels,
                                 dropout=0.1, 
                                 add_self_loops=False,
                                 aggr=aggr)
        self.lin2 = PyG.Linear(-1, embed_channels) # skip connection
        self.norm2 = PyG.norm.BatchNorm(embed_channels) # batch normalization

        # MLP, dim red, regression out
        self.lin3 = PyG.Linear(-1, embed_channels)
        self.lin4 = PyG.Linear(-1, 1)
        self.lin5 = PyG.Linear(-1, 1)
        self.out = PyG.Linear(-1, 1)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index

        # message passing layers w/skip connections, batch norm, and global pooling
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = F.relu(x)
        #x = self.norm1(x)
        x = PyG.pool.glob.global_add_pool(x, batch=batch.batch)

        x = self.conv2(x, edge_index) + self.lin2(x)
        x = F.relu(x)
        #x = self.norm2(x)
        x = PyG.pool.glob.global_add_pool(x, batch=batch.batch)

        # MLP
        x = self.lin3(x)
        x = F.relu(x)
        x = self.lin4(x)
        #x = x.T[0] #transpose and remove empty dimension
        #x = F.pad(x, (0, (400-len(x)))) # pad tensor with zeros, constant length
        x = F.relu(x)
        x = self.lin5(x)
        x = F.relu(x)
        x = self.out(x) 
        return F.logsigmoid(x)