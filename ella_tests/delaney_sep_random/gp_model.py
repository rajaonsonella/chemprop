#------------------------------DATA IMPORTS------------------------------------------
import os
import sys
import pickle

sys.path.append('/home/rajao/chemprop/ella_tests/')
from utils import load_data, merge_tasks, include_test_context
#------------------------------MPCNP/MPNN IMPORTS------------------------------------

import numpy as np
import random

import torch
from torch.utils.tensorboard import SummaryWriter


#------------------------------GP IMPORTS--------------------------------------------

sys.path.append('/home/rajao/FlowMO/')  # to import from GP.kernels and property_predition.data_utils

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
    y_test = torch.cat((y_test,torch.Tensor(t["target_y"][0].type(torch.float)).to(device)),0)
    X_test = torch.cat((X_test,torch.Tensor(t["target_x"][0].type(torch.float)).to(device)),0)

extended_training_set = include_test_context(training_set, test_sets, device)
                                  
#---------------------------------TRAINING PHASE-------------------------------------------

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

