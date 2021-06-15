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

#------------------------------GP IMPORTS--------------------------------------------

sys.path.append('../../../FlowMO/')  # to import from GP.kernels and property_predition.data_utils

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from GP.kernels import Tanimoto
from property_prediction.data_utils import transform_data, TaskDataLoader, featurise_mols

#--------------------------------------INIT------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mpcnp = MPCNP()

#------------------------------------LOAD DATA---------------------------------------------
name = 'Caco2_Wang'
task_name = ['Caco2_Wang']
split_values = (0.8,0.1,0.1)

tasks = load_data(origin='tdc', name=name, task_name=task_name, splitter='random_ksplit', k=None)

train_tasks, valid_tasks, test_tasks = split_tasks(tasks, ratio=split_values, seed=0)

#------------------------------CREATE TEST SETS--------------------------------------------

ratio = 0.2 #Ratio of context/target points in test tasks

smiles_test = []
y_test = torch.Tensor().to(device)
X_test = torch.Tensor().to(device)
test_sets = []

for task in range(len(test_tasks)):
    mpcnp.use_mpnn = False
    test_target_x, test_context_x, test_target_y, test_context_y, test_target_smiles, test_context_smiles = mpcnp.generator(task, test_tasks, context_test_ratio=ratio, is_test=True)
    
    t_set = {"test_target_x": test_target_x,
             "test_context_x": test_context_x,
             "test_target_y": test_target_y,
             "test_context_y": test_context_y,
             "test_target_smiles": test_target_smiles,
             "test_context_smiles": test_context_smiles}
    
    test_sets.append(t_set)
    smiles_test.extend(test_target_smiles[0])
    y_test = torch.cat((y_test,test_target_y),1)
    X_test = torch.cat((X_test,test_target_x),1)

y_test = y_test[-1,:,-1]
X_test = X_test[-1,:,:]

training_set = merge_tasks(train_tasks, device)
validation_set = merge_tasks(valid_tasks, device)

extended_training_set = include_test_context(training_set, test_sets, device)


#DEFINE MPCNP ARGS
mpcnp.epochs  = 50
mpcnp.pred_int = 25
mpcnp.use_mpnn = True

#DEFINE CHEMPROP MODEL ARGS
args = TrainArgs()
args.epochs = mpcnp.epochs
args.dataset_type='regression'
args.task_names=task_name
args.save_dir='chemprop_checkpoints'
args.extra_metrics = ['r2', 'mae']
args.process_args()

data_custom_train = get_data_from_deepchem(extended_training_set["smiles"], extended_training_set["y"])
data_valid = get_data_from_deepchem(validation_set["smiles"], validation_set["y"])
data_test = get_data_from_deepchem(smiles_test, y_test)

#------------------------CHEMPROP TRAINING--------------------------------------------

run_training_deepchem(args, data_custom_train, data_valid, data_test)

#------------------------------GP TRAINING--------------------------------------------

# We define the Gaussian Process Regression training objective

m = None

def objective_closure():
    return -m.log_marginal_likelihood()

gp_training_tasks = [extended_training_set, validation_set]
gp_training_set = merge_tasks(gp_training_tasks, device)

X_train = gp_training_set["X"].cpu()
y_train = gp_training_set["y"].cpu()

X_test = X_test.cpu()

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


_, y_train, _, y_test, y_scaler = transform_data(X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), n_components=50, use_pca=True)

X_train = X_train.type(torch.float64)
X_test = X_test.type(torch.float64)

k = Tanimoto()
m = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(np.mean(y_train)), kernel=k, noise_variance=1)

# Optimise the kernel variance and noise level by the marginal likelihood

opt = gpflow.optimizers.Scipy()
opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))

y_pred, y_var = m.predict_f(X_test)
y_pred = y_scaler.inverse_transform(y_pred)
y_test = y_scaler.inverse_transform(y_test)

gp_r2 = r2_score(y_test, y_pred)
gp_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
gp_mae = mean_absolute_error(y_test, y_pred)

gp_test_metrics = {"gp_r2": gp_r2, "gp_mae": gp_mae, "gp_rmse": gp_rmse}
pickle.dump(gp_test_metrics, open("gp_test_metrics.pkl", "wb"))


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

