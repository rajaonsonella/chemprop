#------------------------------DATA IMPORTS------------------------------------------
import os
import sys
import pickle

sys.path.append('/home/rajao/chemprop/ella_tests')
from utils import load_data, merge_tasks, include_test_context
#------------------------------MPCNP/MPNN IMPORTS------------------------------------

import chemprop
from chemprop.args import TrainArgs
from chemprop.data import get_data_from_deepchem
from chemprop.train import run_training_deepchem

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from model import MPCNP

#--------------------------------------INIT------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_tasks = pickle.load(open("train_tasks.pkl","rb"))
valid_tasks = pickle.load(open("valid_tasks.pkl","rb"))

training_set = merge_tasks(train_tasks, device)
validation_set = merge_tasks(valid_tasks, device)

test_sets = pickle.load(open("test_sets.pkl","rb"))

smiles_test = []
y_test = torch.Tensor().to(device)
X_test = torch.Tensor().to(device)

for t in test_sets:
    smiles_test.extend(t["target_smiles"][0])
    y_test = torch.cat((y_test,t["target_y"][0].type(torch.float).to(device)),0)
    X_test = torch.cat((X_test,t["target_x"][0].type(torch.float).to(device)),0)

extended_training_set = include_test_context(training_set, test_sets, device)

print(y_test.shape)

train_set = get_data_from_deepchem(extended_training_set["smiles"], extended_training_set["y"])
valid_set = get_data_from_deepchem(validation_set["smiles"], validation_set["y"])
test_set = get_data_from_deepchem(smiles_test, y_test)
                                  
#---------------------------------TRAINING PHASE-------------------------------------------

#DEFINE CHEMPROP MODEL ARGS
args = TrainArgs()
args.epochs = 1000
args.dataset_type='regression'
args.task_names= ['expt']
args.save_dir='chemprop_checkpoints'
args.extra_metrics = ['r2', 'mae']
args.process_args()

run_training_deepchem(args, train_set, valid_set, test_set)
