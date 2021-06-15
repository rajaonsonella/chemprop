#------------------------------DATA IMPORTS------------------------------------------
import pickle
import deepchem as dc
from deepchem.feat import CircularFingerprint
from deepchem.splits import ScaffoldSplitter
import os
import sys
import shutil

sys.path.append('../.')
from utils import load_data, split_tasks, merge_tasks, include_test_context, generate_test_sets
#------------------------------------LOAD DATA---------------------------------------------

name = 'delaney-processed'
task_name = ['measured log solubility in mols per litre']
split_values = [(0.8,0.1,0.1)]
ratio = [0.2] #Ratio of context/target points in test tasks

tasks = load_data(origin='deepchem', name=name, task_name=task_name, splitter='random_ksplit', k=None)

#------------------------------------DUMP DATA---------------------------------------------
  
for s in split_values:
    del0=str(s[0]).strip("0.")
    del1=str(s[1]).strip("0.")
    del2=str(s[2]).strip("0.")

    mdir = f'split_{del0}{del1}{del2}' #main directory based on split

    if not os.path.exists(mdir):
        os.makedirs(mdir)

    train_tasks, valid_tasks, test_tasks = split_tasks(tasks, ratio=s, seed=0) #Each is a List[Dict]

    #------------------------------CREATE TEST SETS---------------------------------------------
    
    for r in ratio:
        
        subdir = f'{mdir}/ratio_{str(r).strip("0.")}' #subdirectory based on ratio of context/target points in test tasks

        if not os.path.exists(subdir):
            os.makedirs(subdir)
        
        pickle.dump(train_tasks, open(f"{subdir}/train_tasks.pkl","wb"))
        pickle.dump(valid_tasks, open(f"{subdir}/valid_tasks.pkl","wb"))
            
        test_sets = []
        for task in test_tasks:

            target_x, context_x, target_y, context_y, target_smiles, context_smiles = generate_test_sets(task, context_test_ratio=r, seed=0)

            t_set = {"target_x": target_x,
                     "context_x": context_x,
                     "target_y": target_y,
                     "context_y": context_y,
                     "target_smiles": target_smiles,
                     "context_smiles": context_smiles}

            test_sets.append(t_set)

        pickle.dump(test_sets, open(f"{subdir}/test_sets.pkl","wb"))
        
        #------------------------------COPY MODEL FUNCTIONS-----------------------------------------
        
        shutil.copy("gp_model.py",f"./{subdir}")
        shutil.copy("chemprop_model.py",f"./{subdir}")
        shutil.copy("mpcnp_model.py",f"./{subdir}")
        
        shutil.copy("submit_gp.sh",f"./{subdir}")
        shutil.copy("submit_chemprop.sh",f"./{subdir}")
        shutil.copy("submit_mpcnp.sh",f"./{subdir}")

        

        
