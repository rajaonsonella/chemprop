#------------------------------DATA IMPORTS------------------------------------------
import os
import sys
import pickle

sys.path.append('/home/rajao/chemprop/ella_tests/')
from utils import load_data
#------------------------------MPCNP/MPNN IMPORTS------------------------------------

import chemprop
from chemprop.models import MPN
from chemprop.args import TrainArgs
from chemprop.data import get_data_from_deepchem
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights
from chemprop.train import run_training, run_training_deepchem

import numpy as np
import random

import torch
from torch.utils.tensorboard import SummaryWriter

from model import MPCNP

#--------------------------------------INIT------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mpcnp = MPCNP()

#DEFINE MPCNP ARGS
mpcnp.epochs  = 1000
mpcnp.pred_int = 25
mpcnp.use_mpnn = True

# writer = SummaryWriter('runs_baseline')
writer = None
#---------------------------------TRAINING PHASE-------------------------------------------

train_tasks = pickle.load(open("train_tasks.pkl","rb"))
valid_tasks = pickle.load(open("valid_tasks.pkl","rb"))

mpcnp.train(train_tasks, valid_tasks, writer)

#---------------------------------TESTING PHASE--------------------------------------------

test_sets = pickle.load(open("test_sets.pkl","rb"))

mpcnp_test_metrics = {"r2":[], "mae":[], "rmse":[]}

for t in test_sets: 
    
    context_smiles = t["context_smiles"]
    context_y = t["context_y"]
    
    target_smiles = t["target_smiles"]
    target_y = t["target_y"]
    
    if mpcnp.use_mpnn: 
        # generate the chemprop data loader thing --> context
        context_hs = []
        for batch_ix, (batch_smiles, batch_y) in enumerate(zip(context_smiles, context_y)):
            mol_dataset = get_data_from_deepchem(smiles_array=batch_smiles, targets_array=batch_y)
            graph, y = mol_dataset.batch_graph(), mol_dataset.targets()
            h = mpcnp.mpn(graph)
            context_hs.append(h)
        context_x = torch.stack(context_hs).to(mpcnp.device)

        # generate the target hs
        target_hs = []
        for batch_ix, (batch_smiles, batch_y) in enumerate(zip(target_smiles, target_y)):
            mol_dataset = get_data_from_deepchem(smiles_array=batch_smiles, targets_array=batch_y)
            graph, y = mol_dataset.batch_graph(), mol_dataset.targets()
            h = mpcnp.mpn(graph)
            target_hs.append(h)
        target_x = torch.stack(target_hs).to(mpcnp.device)
    else:
        target_x  = torch.stack(target_x).to(mpcnp.device)            
        context_x = torch.stack(context_x).to(mpcnp.device)            

    # concatenate along a new dimension
    target_y  = torch.stack(target_y).to(mpcnp.device)            
    context_y = torch.stack(context_y).to(mpcnp.device)
    
    #-----PREDICT-----
    
    mu, sigma, dist = mpcnp.predict(context_x, context_y, target_x)
    
    r2, mae, rmse, pearson, spearman = mpcnp.evaluate_prediction(target_y, mu)
    
    mpcnp_test_metrics["r2"].append(r2)
    mpcnp_test_metrics["mae"].append(mae)
    mpcnp_test_metrics["rmse"].append(rmse)
    
mpcnp_test_metrics["r2"] = np.mean(mpcnp_test_metrics["r2"])
mpcnp_test_metrics["mae"] = np.mean(mpcnp_test_metrics["mae"])
mpcnp_test_metrics["rmse"] = np.mean(mpcnp_test_metrics["rmse"])

pickle.dump(mpcnp_test_metrics, open("mpcnp_test_metrics.pkl", "wb"))

if writer is not None:
    writer.flush()
    writer.close()

