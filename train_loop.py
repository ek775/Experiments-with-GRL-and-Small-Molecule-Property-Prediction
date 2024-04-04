#import data
import pandas as pd
from tdc.single_pred import ADME

# import tools for edge splitting
from torch_geometric.transforms import RandomLinkSplit

#load data
data = ADME(name = 'Solubility_AqSolDB')
data = data.get_data()

#import data processing scripts
from smiles_to_tensors import *
#import model
from encoder_model import *

### AutoEncoder
# parameters & data
out_channels = 2

# convert smiles to graphs, tensors
data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(data['Drug'], data['Y'])

#remove molecules with insufficient number of bonds
print("Checking for invalid molecular graphs...")
graphs_before = len(data_list)
data_list = [x for x in data_list if len(x.edge_index[1])>30]
print(f"Removed {(graphs_before-len(data_list))} Invalid Graphs")

# split edges, train/val/test
transform = RandomLinkSplit(
    num_val=0, 
    num_test=0.4,
    is_undirected=True,
    split_labels=True, 
    add_negative_train_samples=False, 
    neg_sampling_ratio=1.0)

train = []
val = []
test = []
print("Splitting Edges...")
for i in data_list:
    train_data, val_data, test_data = transform(i)
    train.append(train_data)
    val.append(val_data)
    test.append(test_data)

# initialize dataloaders
train_dataloader = DataLoader(dataset=train, batch_size=32, shuffle=True)
val_dataloader = DataLoader(dataset=val, batch_size=1, shuffle=True)
test_dataloader = DataLoader(dataset=test, batch_size=1, shuffle=True)

# initialize model
num_features = data_list[0].num_features
model = GAE(GCNEncoder(num_features, out_channels)) # GAE default decoder is inner dot product
print(model.parameters)

# loss fn = GAE built in reconstruction loss (Kipf and Welling, 2016)

# move to GPU (if available)
device = 'cpu' #stay on cpu for now...configuration issues #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

### TRAINING ###

def train_loop(dataloader, model, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    total_loss=0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = batch.to(device)
        #print(f'Graph G: {batch}')
        # Compute prediction and loss
        z = model.encode(batch.x, batch.edge_index)
        #print(f'Embedded G as: {z}')
        loss = model.recon_loss(z, batch.edge_index)
        # Backpropagation
        loss.backward()
        optimizer.step()
        #aggregate loss
        total_loss+=loss
    return float(loss)

def test(val_data, model):
    model.eval()
    with torch.no_grad():
        auc_total = []
        ap_total = []
        for i in val_data:
            z = model.encode(i.x, i.edge_index)
            auc, ap = model.test(z, pos_edge_index=i.pos_edge_label_index, neg_edge_index=i.neg_edge_label_index)
            auc_total.append(auc)
            ap_total.append(ap)
        auc_avg = sum(auc_total)/len(auc_total)
        ap_avg = sum(ap_total)/len(ap_total)
    return auc_avg, ap_avg

# Training and Eval
epochs = 100
history = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    epoch_loss = train_loop(dataloader=train_dataloader, model=model, optimizer=optimizer)
    auc, ap = test(val_data=test_dataloader, model=model)
    results = [('Loss:', epoch_loss),('AUC:', auc),('Avg Prec', ap)]
    print(results)
    history.append(results)
print("Done!")

#write history to csv
print("===============================")
print("Saving Model")
torch.save(model, f="model_30edgemin_dLR")
print("===============================")

print("Writing Training History...")
history = pd.DataFrame(history)
history.to_csv("./train_hist_4")
print("===COMPLETE===")