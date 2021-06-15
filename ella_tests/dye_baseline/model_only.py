#------------------------------DATA IMPORTS------------------------------------------
import pickle
import deepchem as dc
from deepchem.feat import CircularFingerprint
from deepchem.splits import ScaffoldSplitter
import os
import sys

sys.path.append('../.')
from utils import load_data, split_tasks, merge_tasks, include_test_context
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
import GPUtil, gc

#--------------------------------------INIT------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mpcnp = MPCNP()

#------------------------------------LOAD DATA---------------------------------------------
name = 'dye_lasers_marg'
task_name = ['fluo_peak_1']
split_values = (0.8,0.1,0.1)

tasks = load_data(origin='pkl_file', name=name, task_name=task_name, splitter=None, k=None)

train_tasks, valid_tasks, test_tasks = split_tasks(tasks, ratio=split_values, seed=0)

#------------------------------CREATE TEST SETS--------------------------------------------

ratio = 0.2 #Ratio of context/target points in test tasks

#DEFINE MPCNP ARGS
mpcnp.epochs  = 1000
mpcnp.pred_int = 25
mpcnp.use_mpnn = True

#---------------------------MPCNP TRAINING--------------------------------------------
mpcnp.use_mpnn = True
writer = SummaryWriter('runs_baseline/{}'.format(name))
mpcnp.train(train_tasks, valid_tasks, writer)

test_sets = []
for task in range(len(test_tasks)):
    mpcnp.use_mpnn = True
    test_target_x, test_context_x, test_target_y, test_context_y, test_target_smiles, test_context_smiles = mpcnp.generator(task, test_tasks, context_test_ratio=ratio, is_test=True)

    
    t_set = {"test_target_x": test_target_x,
             "test_context_x": test_context_x,
             "test_target_y": test_target_y,
             "test_context_y": test_context_y,
             "test_target_smiles": test_target_smiles,
             "test_context_smiles": test_context_smiles}
    test_sets.append(t_set)

for t_set in test_sets:
    mpcnp.use_mpnn = True
    mu, sigma, dist = mpcnp.predict(t_set["test_context_x"], t_set["test_context_y"], t_set["test_target_x"])
    test_r2, test_mae, test_rmse, test_pearson, test_spearman = mpcnp.evaluate_prediction(t_set["test_target_y"], mu)
    
mpcnp_test_metrics = {"mpcnp_r2": test_r2, "mpcnp_mae": test_mae, "mpcnp_rmse": test_rmse}
pickle.dump(mpcnp_test_metrics, open("mpcnp_test_metrics.pkl", "wb"))

writer.flush()
writer.close()

