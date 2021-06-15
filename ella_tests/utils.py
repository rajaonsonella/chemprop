from typing import List, Union, Dict

import random
import numpy as np
import torch

import deepchem as dc 
from deepchem import molnet
from deepchem.splits import ScaffoldSplitter, RandomSplitter
from deepchem.feat import CircularFingerprint
from deepchem.molnet.load_function.molnet_loader import featurizers, splitters, transformers, TransformerGenerator, _MolnetLoader

import tdc
from tdc.single_pred import ADME

import rdkit

import os
import math
import pickle
import pandas as pd

#-----------------------------------DATA LOADING------------------------------------------

def load_data(origin, name, task_name, splitter=None, k=None, seed=None):
    '''
    Loads data from various databases. Databases implemented so far:
        - MolNet (Deepchem):
            format supported: csv files (offline)
        - TDC:
            format supported: tab files (offline)
        
    origin: str - name of the database ('deepchem'/'tdc')
    name: str - name of the dataset. If CSV, must match file name.
    task_name: List[str] - name of the task. If CSV, must match column name.
    splitter: str - name of data-splitting methods. Implemented so far:
                    ---ALL Databases---
                    None: returns the plain Dataset -> Dict 
                    scaffold_tasks: returns the Murcko scaffold sets as Tasks -> List[Dict]
                    scaffold_ksplit: returns k sets formed using Murcko scaffolding as Tasks -> List[Dict]
                    
                    random_ksplit: returns k sets formed randomly as Tasks -> List[Dict]
                    -------MolNet------
                    scaffold: Uses Murcko scaffolding, outputs Train/Valid/Test Dataset -> Dict[Dict]
                    fingerprint: Uses ECFP fingerprints, outputs Train/Valid/Test Dataset -> Dict[Dict]
                    --------TDC--------
                    #TODO
    k: int - number of sets if "scaffold_ksplit" or "random_ksplit", default: k=10
    
    return: Dataset -> Dict
            Train/Valid/Test split Dataset -> Dict[Dict]
            Tasks -> List[Dict]
    '''
    
    if origin == 'deepchem':
    
        if splitter != None:

            if splitter == 'scaffold_tasks':
                
                print("Selected {} splitting option, will return Tasks in the form: List[Dict]".format(splitter))

                lder = OfflineLoader(featurizer='ECFP', 
                         splitter=None, 
                         transformer_generators=['normalization'],
                         tasks=task_name,
                         data_dir='../../../deepchem-data',
                         save_dir = None)

                task, data, ids = lder.load_dataset(name, reload=False)

                ssplitter = ScaffoldSplitter()
                scaffold_sets = ssplitter.generate_scaffolds(data[0])

                filt_scaffold_sets = filter_scaffold_sets(scaffold_sets)

                tasks = []
                for sset in filt_scaffold_sets:

                    task_smiles = data[0].ids[sset]
                    task_X      = data[0].X[sset]
                    task_y      = data[0].y[sset]

                    tasks.append({'smiles':task_smiles.tolist(), 
                                      'X':torch.from_numpy(task_X), 
                                      'y':torch.from_numpy(task_y)}
                                )

                return tasks
            
            if splitter == 'scaffold_ksplit':
                
                print("Selected {} splitting option, will return Tasks in the form: List[Dict]".format(splitter))
                
                lder = OfflineLoader(featurizer='ECFP', 
                         splitter=None, 
                         transformer_generators=['normalization'],
                         tasks=task_name,
                         data_dir='../../../deepchem-data',
                         save_dir = None)

                task, data, ids = lder.load_dataset(name, reload=False)
                
                ssplitter = ScaffoldSplitter()
                if k == None:
                    print("No specified k, switching to default: k=10")
                    k=10
                
                tasks_tuple = ssplitter.k_fold_split(data[0], k)

                tasks = []
                for t, f in tasks_tuple:
                    X = f.X        # ECFP fingerprints
                    y = f.y        # target
                    X = torch.Tensor(X)
                    y = torch.Tensor(y)
                    smiles = f.ids # smiles

                    dataset = {'smiles': smiles, 'X': X, 'y': y}
                    tasks.append(dataset)

                return tasks
            
            if splitter == 'random_ksplit':
                
                print("Selected {} splitting option, will return Tasks in the form: List[Dict]".format(splitter))
                
                lder = OfflineLoader(featurizer='ECFP', 
                         splitter=None, 
                         transformer_generators=['normalization'],
                         tasks=task_name,
                         data_dir='../../../deepchem-data',
                         save_dir = None)

                task, data, ids = lder.load_dataset(name, reload=False)
                
                ssplitter = RandomSplitter()
                if k == None:
                    print("No specified k, switching to default: k=10")
                    k=10
                
                tasks_tuple = ssplitter.k_fold_split(data[0], k)
                
                tasks = []
                for t, f in tasks_tuple:
                    X = f.X        # ECFP fingerprints
                    y = f.y        # target
                    X = torch.Tensor(X)
                    y = torch.Tensor(y)
                    smiles = f.ids # smiles

                    dataset = {'smiles': smiles, 'X': X, 'y': y}
                    tasks.append(dataset)

                return tasks

            else:

                print("Selected {} splitting option, will return a Train/Valid/Test Dataset in the form: Dict[Dict]".format(splitter))

                lder = OfflineLoader(featurizer='ECFP', 
                         splitter=splitter, 
                         transformer_generators=['normalization'],
                         tasks=task_name,
                         data_dir='../../../deepchem-data',
                         save_dir = None)

                task, data, ids = lder.load_dataset(name, reload=False)

                print("Loading the {} dataset".format(name))

                X_train = data[0].X        # ECFP fingerprints
                y_train = data[0].y        # targets
                smiles_train = data[0].ids # smiles

                print("Training set characteristics:", X_train.shape, y_train.shape, len(smiles_train))

                train = {'smiles': smiles_train, 'X': X_train, 'y': y_train}

                X_valid = data[1].X        
                y_valid = data[1].y       
                smiles_valid = data[1].ids 

                print("Validation set characteristics:", X_valid.shape, y_valid.shape, len(smiles_valid))

                valid = {'smiles': smiles_valid, 'X': X_valid, 'y': y_valid}

                X_test = data[2].X        
                y_test = data[2].y       
                smiles_test = data[2].ids

                print("Testing set characteristics:", X_test.shape, y_test.shape, len(smiles_test))

                test = {'smiles': smiles_test, 'X': X_test, 'y': y_test}

                dataset = {'train': train, 'valid': valid, 'test': test}

                return dataset

        else:

            print("No selected splitting, will return a Dataset in the form: Dict")

            lder = OfflineLoader(featurizer='ECFP', 
                         splitter=None, 
                         transformer_generators=['normalization'],
                         tasks=task_name,
                         data_dir='../../../deepchem-data',
                         save_dir = None)

            task, data, ids = lder.load_dataset(name, reload=False)

            X = data[0].X        # ECFP fingerprints
            y = data[0].y        # targets
            smiles = data[0].ids # smiles

            print("Loading the {} dataset".format(name))
            print("Characteristics:", X.shape, y.shape, len(smiles))

            dataset = {'smiles': smiles, 'X': X, 'y': y}

            return dataset
    
    if origin == 'tdc':
    
        feat = CircularFingerprint()

        data = ADME(name)
        df = data.get_data()
        ids = df["Drug"].to_numpy()
        y = df["Y"].to_numpy()
        y = y.reshape(-1,1)
        X = feat(ids)

        dataset = dc.data.DiskDataset.from_numpy(X=X, y=y, ids=ids, tasks=task_name)
        
        if splitter != None:
            
            if splitter == 'scaffold_tasks':
                
                print("Selected {} splitting option, will return Tasks in the form: List[Dict]".format(splitter))
                
                ssplitter = ScaffoldSplitter()
                scaffold_sets = ssplitter.generate_scaffolds(dataset)

                filt_scaffold_sets = filter_scaffold_sets(scaffold_sets)

                tasks = []
                for sset in filt_scaffold_sets:

                    task_smiles = data[0].ids[sset]
                    task_X      = data[0].X[sset]
                    task_y      = data[0].y[sset]

                    tasks.append({'smiles':task_smiles.tolist(), 
                                      'X':torch.from_numpy(task_X), 
                                      'y':torch.from_numpy(task_y)}
                                )

                return tasks
            
            if splitter == 'scaffold_ksplit':
                
                print("Selected {} splitting option, will return Tasks in the form: List[Dict]".format(splitter))
                
                ssplitter = ScaffoldSplitter()
                if k == None:
                    print("No specified k, switching to default: k=10")
                    k=10
                
                tasks_tuple = ssplitter.k_fold_split(dataset, k)

                tasks = []
                for t, f in tasks_tuple:
                    X = f.X        # ECFP fingerprints
                    y = f.y        # target
                    X = torch.Tensor(X)
                    y = torch.Tensor(y)
                    smiles = f.ids # smiles

                    dataset = {'smiles': smiles, 'X': X, 'y': y}
                    tasks.append(dataset)

                return tasks
            
            if splitter == 'random_ksplit':
                
                print("Selected {} splitting option, will return Tasks in the form: List[Dict]".format(splitter))
                
                ssplitter = RandomSplitter()
                if k == None:
                    print("No specified k, switching to default: k=10")
                    k=10
                
                tasks_tuple = ssplitter.k_fold_split(dataset, k)

                tasks = []
                for t, f in tasks_tuple:
                    X = f.X        # ECFP fingerprints
                    y = f.y        # target
                    X = torch.Tensor(X)
                    y = torch.Tensor(y)
                    smiles = f.ids # smiles

                    dataset = {'smiles': smiles, 'X': X, 'y': y}
                    tasks.append(dataset)

                return tasks

        
        else:
            print("No selected splitting, will return a Dataset in the form: Dict")
        
            X = dataset.X        # ECFP fingerprints
            y = dataset.y        # targets
            smiles = dataset.ids # smiles

            print("Loading the {} dataset".format(name))
            print("Characteristics:", X.shape, y.shape, len(smiles))

            dataset = {'smiles': smiles, 'X': X, 'y': y}
            
            return dataset

    if origin == 'pkl_file':
        
        feat = CircularFingerprint()
        dataset = pickle.load(open(f"{name}.pkl","rb"))
        df = pd.DataFrame()
        tasks = []
        for k in dataset.keys():
            df = dataset[k]["df"]
            ids = df["smiles"].to_numpy()
            y = df[f"{task_name[0]}"].to_numpy()
            y = y.reshape(-1,1)
            X = feat(ids)

            task = {'smiles': ids, 'X': torch.from_numpy(X), 'y': torch.from_numpy(y)}
            tasks.append(task)
        return tasks

            #dataset = dc.data.DiskDataset.from_numpy(X=X, y=y, ids=ids, tasks=task_name)
        
        
        
        
def split_tasks(tasks, ratio, seed=None):
    
    np.random.seed(seed)
    np.random.shuffle(tasks)
    
    num_train, num_valid, num_test = ratio
    
    nb_tasks = len(tasks)
    split_1 = round(num_train*nb_tasks)
    split_2 = round((num_train+num_valid)*nb_tasks)

    train_tasks = tasks[:split_1]
    valid_tasks = tasks[split_1:split_2]
    test_tasks = tasks[split_2:]

    return train_tasks, valid_tasks, test_tasks
        
def split_into_tasks(dataset, num_tasks, num_per_task):
    np.random.seed()
    
    smiles = dataset['smiles'] 
    X = dataset['X']
    y =  dataset['y']
    
    tasks = []
    if isinstance(num_per_task, list):
        if len(num_per_task) != num_tasks:
            raise ValueError('Length of num_per_task must match num_tasks.')
            
        if sum(num_per_task) != X.shape[0]:
            raise ValueError('Sum of elements in num_per_task must match number of elements in dataset.')
        
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        
        tasks = []
        for num in num_per_task:

            task_smiles = smiles[indices[:num]]
            task_X      = X[indices[:num], :]
            task_y      = y[indices[:num], :]

            tasks.append({'smiles':task_smiles.tolist(), 
                          'X':torch.from_numpy(task_X), 
                          'y':torch.from_numpy(task_y)}
                    )
            
            indices = np.roll(indices, num)      
        
    else: 
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)  
        for task_ix in range(num_tasks):

            task_smiles = smiles[indices[:num_per_task]]
            task_X      = X[indices[:num_per_task], :]
            task_y      = y[indices[:num_per_task], :]

            tasks.append({'smiles':task_smiles.tolist(), 
                          'X':torch.from_numpy(task_X), 
                          'y':torch.from_numpy(task_y)}
                    )

            indices = np.roll(indices, num_per_task)
  
    return tasks


def filter_scaffold_sets(scaffold_sets):
    '''Exclude scaffold sets with too few points'''
    filtered_sets = []
    for sets in scaffold_sets:
    
        if len(sets) < 4:
            continue
            
        filtered_sets.append(sets)
    return filtered_sets


class OfflineLoader(_MolnetLoader):
    '''Special MolNet DataLoader adapted to cluster (no internet access)'''
    def create_dataset(self, name):
        dataset_file = os.path.join(self.data_dir, f"{name}.csv")
        
        if not os.path.exists(dataset_file):
          print("Path to file does not exist")
          dc.utils.data_utils.download_url(url=DELANEY_URL, dest_dir=self.data_dir)
        
        loader = dc.data.CSVLoader(
            tasks=self.tasks, feature_field="smiles", featurizer=self.featurizer)
        return loader.create_dataset(dataset_file, shard_size=8192)
    
    def load_dataset(self, name, reload: bool):

        # Build the path to the dataset on disk.

        featurizer_name = str(self.featurizer)
        splitter_name = 'None' if self.splitter is None else str(self.splitter)
        save_folder = os.path.join(self.save_dir, name + "-featurized",
                                   featurizer_name, splitter_name)
        if len(self.transformers) > 0:
          transformer_name = '_'.join(
              t.get_directory_name() for t in self.transformers)
          save_folder = os.path.join(save_folder, transformer_name)

        # Try to reload cached datasets.

        if reload:
          if self.splitter is None:
            if os.path.exists(save_folder):
              transformers = dc.utils.data_utils.load_transformers(save_folder)
              return self.tasks, (DiskDataset(save_folder),), transformers
          else:
            loaded, all_dataset, transformers = dc.utils.data_utils.load_dataset_from_disk(
                save_folder)
            if all_dataset is not None:
              return self.tasks, all_dataset, transformers

        # Create the dataset

        dataset = self.create_dataset(name)

        # Split and transform the dataset.

        if self.splitter is None:
          transformer_dataset = dataset
        else:
          train, valid, test = self.splitter.train_valid_test_split(dataset)
          transformer_dataset = train
        transformers = [
            t.create_transformer(transformer_dataset) for t in self.transformers
        ]
        if self.splitter is None:
          for transformer in transformers:
            dataset = transformer.transform(dataset)
          if reload and isinstance(dataset, DiskDataset):
            dataset.move(save_folder)
            dc.utils.data_utils.save_transformers(save_folder, transformers)
          return self.tasks, (dataset,), transformers

        for transformer in transformers:
          train = transformer.transform(train)
          valid = transformer.transform(valid)
          test = transformer.transform(test)
        if reload and isinstance(train, DiskDataset) and isinstance(
            valid, DiskDataset) and isinstance(test, DiskDataset):
          dc.utils.data_utils.save_dataset_to_disk(save_folder, train, valid, test,
                                                   transformers)
        return self.tasks, (train, valid, test), transformers

#-----------------------------------BASELINE DATA HANDLING--------------------------------

def merge_tasks(tasks_list, device):
    merged_tasks = {"smiles":[], "X": torch.Tensor().to(device), "y": torch.Tensor().to(device)}
    for task in tasks_list:
        merged_tasks["smiles"].extend(task["smiles"])
        merged_tasks["X"] = torch.cat((merged_tasks["X"],task["X"].to(device)),0)
        merged_tasks["y"] = torch.cat((merged_tasks["y"],task["y"].to(device)),0)
    return merged_tasks

#Extend training_set with test_context
def include_test_context(training_set, test_sets, device):
    extended_training_set = {"smiles": [], "X": torch.Tensor().to(device), "y": torch.Tensor().to(device)}
    extended_training_set["smiles"] = training_set["smiles"]
    extended_training_set["X"] = training_set["X"].to(device)
    extended_training_set["y"] = training_set["y"].to(device)
    for t_set in test_sets:
            extended_training_set["smiles"].extend(t_set["context_smiles"][0])
            extended_training_set["X"]  = torch.cat((extended_training_set["X"],t_set["context_x"][0].type(torch.float).to(device)),0).to(device)
            extended_training_set["y"]  = torch.cat((extended_training_set["y"],t_set["context_y"][0].type(torch.float).to(device)),0).to(device)
    return extended_training_set

def generate_test_sets(task, context_test_ratio=None, seed=None):
    
    np.random.seed(seed)

    smiles = np.array(task['smiles'])
    x      = task['X']
    y      = task['y']

    if context_test_ratio == None:
        print("No context/target sets ratio specified for test, switching to default: 0.2 ")
        context_test_ratio = 0.2 

    num_context = int(context_test_ratio*x.shape[0])
    num_target = x.shape[0] - num_context

    indices = np.random.permutation(x.shape[0])

    target_x  = [x[indices[num_context:], :]]
    target_y  = [y[indices[num_context:], :]]

    target_smiles = [smiles[indices[num_context:]]]

    context_x = [x[indices[:num_context], :]]
    context_y = [y[indices[:num_context], :]]

    context_smiles = [smiles[indices[:num_context]]]

    return target_x, context_x, target_y, context_y, target_smiles, context_smiles
