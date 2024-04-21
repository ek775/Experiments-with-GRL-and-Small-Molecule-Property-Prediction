# importing utilities and data
from smiles_to_tensors import *
from attn_graph_forward_sol import *
import pandas as pd
from tdc.single_pred import ADME
from torch.utils.tensorboard import SummaryWriter

# load & split data 
data = ADME(name = 'Solubility_AqSolDB')
data = data.get_split(frac=[0.8,0.1,0.1])
train_df, val_df, test_df = data['train'], data['valid'], data['test']
train_data = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(train_df['Drug'], train_df['Y'])
val_data = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(val_df['Drug'], val_df['Y'])
test_data = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(test_df['Drug'], test_df['Y'])

# create dataloaders
train_dl = DataLoader(train_data, shuffle=True)
val_dl = DataLoader(val_data, shuffle=True)

# initialize model
num_features = train_data[0].num_features
model = GACsol(in_channels=num_features)
print(model.parameters)

# move to GPU (if available)
device = 'cpu' #stay on cpu for now...configuration issues #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, maximize=True)





### FUNCTIONS ###

def train_loop(dataloader, model, optimizer):
    # Set the model to training mode
    model.train()
    total_loss=0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = batch.to(device)
        print(batch) # debug me
        # Compute prediction and loss
        pred_log_sol = model(batch)
        print(pred_log_sol) # debug me
        true_log_sol = batch.y
        loss = torch.nn.MSELoss(pred_log_sol,true_log_sol)
        # Backpropagation
        loss.backward()
        optimizer.step()
        #aggregate loss
        total_loss+=loss
    return float(loss)

def validate(dataloader, model):
    model.eval()
    with torch.no_grad():
        preds = []
        targets = []
        for i in dataloader:
            preds.append(model(i)[0])
            targets.append(i.y[0])
        loss = torch.nn.MSELoss(preds,targets)
        return preds, targets, loss





### TRAINING AND EVAL ###

history = SummaryWriter('GATv2Conv')
epochs = 100
import time
# begin training
for e in range(1, epochs+1):
    print(f"=====|Epoch {e}|=====")
    start = time.time()
    #train, backprop, validation_metrics
    train_loss = train_loop(dataloader=train_dl, model=model, optimizer=optimizer)
    val_loss, val_preds, val_targets = validate(val_dl, model)
    val_error = [val_preds[i]-val_targets[i] for i in range(len(val_targets))]
    stop = time.time()
    # record training
    history.add_scalars(tag='Training & Validation Loss', 
                        tag_scalar_dict={'train':train_loss,
                                         'val':val_loss}, 
                        global_step=e)
    history.add_scalars(tag='Raw Validation Error', 
                        tag_scalar_dict={'pred_sol':val_preds, 
                                         'target_sol':val_targets, 
                                         'raw_error':val_error}, 
                        global_step=e)
    # show progress in terminal
    print(f"train loss: {train_loss} | val loss: {val_loss} | raw error: {val_error} | Time: {stop-start}s")

#save history
history.close()