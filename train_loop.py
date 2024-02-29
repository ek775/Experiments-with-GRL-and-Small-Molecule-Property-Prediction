#import data
import pandas as pd
from tdc.single_pred import ADME

#load and train-test split (0.7, 0.1, 0.2 by default)
data = ADME(name = 'Solubility_AqSolDB')
data = data.get_split()
train = data['train']
val = data['valid']
test = data['test']

#import packages

#import data processing scripts
from smiles_to_tensors import *
#import model
from encoder_model import *