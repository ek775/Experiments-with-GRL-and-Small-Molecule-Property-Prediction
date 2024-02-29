#import data
import pandas as pd
from tdc.single_pred import ADME

#load and train-test split (0.7, 0.1, 0.2 by default)
data = ADME(name = 'Solubility_AqSolDB')
data = data.get_split()
train = data['train']
val = data['valid']
test = data['test']

#import data processing scripts
from smiles_to_tensors import *
#import model
from encoder_model import *

# AutoEncoder
# parameters & data
out_channels = 2

train_data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(train['Drug'], train['Y'])
train_data_list = [train_test_split_edges(i) for i in train_data_list] # split edges for reconstruction loss
print(train_data_list[0].__getattr__)
train_dataloader = DataLoader(dataset=train_data_list, batch_size=1)

num_features = train_data_list[0].num_features

# model
model = GAE(GCNEncoder(num_features, out_channels)) # GAE default decoder is inner dot product

# loss fn = GAE built in reconstruction loss (Kipf and Welling, 2016)

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

### TRAINING ###

def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for (k, batch) in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        # Compute prediction and loss
        z = model.encode(batch.x, batch.train_pos_edge_index)
        loss = model.recon_loss(z, batch.train_pos_edge_index)
        # Backpropagation
        loss.backward()
        optimizer.step()
        print(f"Reconstruction-Loss: {loss}")

# move fast and break things
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, optimizer)
print("Done!")