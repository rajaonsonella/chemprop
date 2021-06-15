from typing import List, Union, Dict
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

import chemprop
from chemprop.models import MPN
from chemprop.args import TrainArgs
from chemprop.data import get_data_from_deepchem

from metrics import Metrics

# MLP
class MLPEncoder(nn.Module):
    ''' Densely connected feed-forward MLP - encoder 
    '''
    def __init__(self,
                 params,
                 inp_size,
                 **kwargs
        ):
        super(MLPEncoder, self).__init__()
        self.params = params
        self.inp_size = inp_size
        
        num_layers = len(self.params['hidden_size'])+1
        
        layers = []
        for hidden_ix, hidden_size in enumerate(self.params['hidden_size']):
            if hidden_ix == 0:
                layers.append(nn.Linear(inp_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self.params['hidden_act'])
            
        layers.append(nn.Linear(self.params['hidden_size'][-1], self.params['out_size']))
        #layers.append(self.params['out_act'])
        
        self.mlp = nn.Sequential(*layers)

        
        
    def forward(self, 
                mpn_rep_x,
                context_y,
            ) -> torch.FloatTensor:
        ''' forward pass of fully-connected MLP
        
        Args:
            mpn_rep_x : output of mpnn for context_x
            context_y : respective context_y values
        '''
        
        # concatenate the mpn_rep_x and context_y
        enc_input = torch.cat((mpn_rep_x, context_y), dim=-1) # [batch_size, n_context, inp_size]
        batch_size, filter_size = enc_input.shape[0], enc_input.shape[-1]                       
        enc_input = enc_input.view(-1, self.inp_size)             # [batch_size * n_context, inp_size]
        enc_out = self.mlp(enc_input.float())                         # [batch_size * n_context, hidden_size]
        
        enc_out = enc_out.view(batch_size, -1, self.params['out_size']) # [batch_size, n_context, hidden_size]
        rep     = torch.mean(enc_out, 1)                      # [batch_size, hidden_size]
        
        return rep

# MLP
class MLPDecoder(nn.Module):
    ''' Densely connected feed-forward MLP - encoder 
    '''
    def __init__(self,
                 params,
                 inp_size,
                 **kwargs
        ):
        super(MLPDecoder, self).__init__()
        self.params = params
        self.inp_size = inp_size
        
        num_layers = len(self.params['hidden_size'])+1
        
        layers = []
        for hidden_ix, hidden_size in enumerate(self.params['hidden_size']):
            if hidden_ix == 0:
                layers.append(nn.Linear(inp_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self.params['hidden_act'])
            
        layers.append(nn.Linear(self.params['hidden_size'][-1], self.params['out_size']))
        #layers.append(self.params['out_act'])
        
        self.mlp = nn.Sequential(*layers)

        
        
    def forward(self, 
                mpn_rep_x, 
                enc_rep,
            ) -> torch.FloatTensor:
        ''' forward pass of fully-connected MLP
        
        Args:
            mpn_rep_x : output of mpnn for target_x
            enc_rep   : representation from encoder
        '''

        
        batch_size = mpn_rep_x.shape[0]
        
        enc_rep = torch.unsqueeze(enc_rep, dim=1).repeat(1, mpn_rep_x.shape[1], 1)  # inflate inputs
        
        # filter_size = enc_rep_size + mpn_rep_x_size
        dec_input = torch.cat((mpn_rep_x, enc_rep), dim=2)                  # [batch_size, n_target, filter_size]
    
        dec_input = dec_input.view(-1, self.inp_size)                       # [batch_size*n_target, filter_size]

        dec_out = self.mlp(dec_input.float())                               # [batch_size*n_target, 2*output_size] - Ella: Do you mean: 2*self.y_size ?
        
        mu, log_sigma = torch.split(dec_out, 1, dim=1)
        
        # TODO:  last dim is hardcoded - fix this
        mu = mu.view(batch_size, -1, 1)                                     # [batch_size, n_target, output_size]
        
        sigma = 0.1 + 0.9 * nn.Softplus()(log_sigma)
        sigma = sigma.view(batch_size, -1, 1)
        
        dist = [MultivariateNormal(m, torch.diag_embed(s)) for m, s in zip(mu, sigma)]
        
        return mu, sigma, dist


# MPCNP model
class MPCNP():
    ''' Message-passing conditional neural process model
    '''
    DEF_ARGS = TrainArgs()
    DEF_ARGS.mpn_shared = True
    DEF_ARGS.dataset_type = 'regression'
    DEF_ARGS.process_args()
    
    DEF_ENCODER_PARAMS   = {'hidden_size': [100, 100],
                            'out_size': 50, 
                            'hidden_act': nn.ReLU(), #lambda y: nn.ReLU(y),
                    }
    DEF_DECODER_PARAMS   = {'hidden_size': [100, 100],
                            'out_size': 2, 
                            'hidden_act': nn.ReLU(),
                    }
 

    def __init__(self,
                 args: TrainArgs=None,               # MPNN args
                 encoder_params:Dict[str, any]=None,
                 decoder_params:Dict[str, any]=None,
                 use_mpnn:bool=True, 
                 learning_rate:float=1e-4,
                 batch_size:int=20, 
                 pred_int:int=25, 
                 epochs:int=8000, 
                 x_size:int=300,
                 y_size:int=1,
                 context_x_scaling:str='same',
                 context_y_scaling:str='same', 
                 target_x_scaling:str='same',
                 target_y_scaling:str='same', 
                 device:str='gpu',
                 writer: SummaryWriter = None,
                 **kwargs,
    ):
#        super(MPCNP, self).__init__()
        self.use_mpnn = use_mpnn
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.pred_int = pred_int
        self.epochs = epochs
        self.x_size = x_size           # this should be embedding size in this case
        self.y_size = y_size
        self.output_size = 2*self.y_size # mean + var for each y dim
        self.context_x_scaling = context_x_scaling
        self.context_y_scaling = context_y_scaling
        self.target_x_scaling = target_x_scaling
        self.target_y_scaling = target_y_scaling
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'DEVICE : {self.device}')

        self.encoder_params = self._parse_params(
                                    encoder_params,
                                    self.DEF_ENCODER_PARAMS,
                            )
        self.decoder_params = self._parse_params(
                                    decoder_params, 
                                    self.DEF_DECODER_PARAMS,
                            )
        
        self._create_encoder()
        params_encoder = list(self.encoder.parameters())
        self._create_decoder()
        params_decoder = list(self.decoder.parameters())
        
        
        if self.use_mpnn:
            self.args = args or self.DEF_ARGS
            self._create_mpn()
            params_mpnn = list(self.mpn.parameters())
        else:
            params_mpnn = []
            
        
        self.optimizer = torch.optim.Adam(
                                    params_mpnn+
                                    params_encoder+
                                    params_decoder,
                                    lr=self.learning_rate,
                    )
        
        self.metrics = Metrics()
        
    
   
    def _parse_params(self, user_params, default_params):
        if not user_params:
            params = default_params
        
        elif type(user_params)==dict:
            params = {}
            for key, val in default_params.items():
                try:
                    params[key] = user_params[key] 
                except: 
                    params[key] = val
        else:
            print('@@\tUser hyperparams not understood, resorting to default... [WARNING] ')
            params = default_params
        return params
            
        
    def _create_mpn(self):
        self.mpn = MPN(self.args)
        self.mpn.to(self.device)
    
    def _create_encoder(self):
        self.encoder = MLPEncoder(self.encoder_params, self.x_size+self.y_size)
        self.encoder.to(self.device)
        
    def _create_decoder(self):
        self.decoder = MLPDecoder(self.decoder_params, self.encoder_params['out_size']+self.x_size)
        self.decoder.to(self.device)
        
    def loss_fn(self, dist, target_y):
        ''' negative log-likelihood
        '''
        log_probs = [d.log_prob(target_y[_, ...].float()) for _, d in enumerate(dist)]
        return -torch.mean(torch.cat(log_probs, dim=0))
    
    def generator(self, num_tasks, all_tasks):
            
        task_ix = np.random.randint(num_tasks)

        task = all_tasks[task_ix]

        smiles = np.array(task['smiles'])
        x      = task['X']
        y      = task['y']

        num_context = torch.randint(low=int(0.2*x.shape[0]), high=int(0.8*x.shape[0]), size=(1,))
        num_target = x.shape[0] - num_context

        indices = [np.random.permutation(x.shape[0]) for _ in range(self.batch_size)]

        target_x  = [x[indices[_][num_context:], :] for _ in range(self.batch_size)]
        target_y  = [y[indices[_][num_context:], :] for _ in range(self.batch_size)]

        target_smiles = [smiles[indices[_][num_context:]] for _ in range(self.batch_size)]

        context_x = [x[indices[_][:num_context], :] for _ in range(self.batch_size)]
        context_y = [y[indices[_][:num_context], :] for _ in range(self.batch_size)]

        context_smiles = [smiles[indices[_][:num_context]] for _ in range(self.batch_size)]

        if self.use_mpnn:                 
            # generate the chemprop data loader thing --> context
            context_hs = []
            for batch_ix, (batch_smiles, batch_y) in enumerate(zip(context_smiles, context_y)):
                mol_dataset = get_data_from_deepchem(smiles_array=batch_smiles, targets_array=batch_y)
                graph, y = mol_dataset.batch_graph(), mol_dataset.targets()
                h = self.mpn(graph)
                context_hs.append(h)
            context_x = torch.stack(context_hs).to(self.device)          

            # generate the target hs
            target_hs = []
            for batch_ix, (batch_smiles, batch_y) in enumerate(zip(target_smiles, target_y)):
                mol_dataset = get_data_from_deepchem(smiles_array=batch_smiles, targets_array=batch_y)
                graph, y = mol_dataset.batch_graph(), mol_dataset.targets()
                h = self.mpn(graph)
                target_hs.append(h)
            target_x = torch.stack(target_hs).to(self.device)      

        else:
            target_x  = torch.stack(target_x).to(self.device)            
            context_x = torch.stack(context_x).to(self.device)            

        # concatenate along a new dimension
        target_y  = torch.stack(target_y).to(self.device)            
        context_y = torch.stack(context_y).to(self.device)             
    
        return target_x, context_x, target_y, context_y, target_smiles, context_smiles

        
    def train(self,
              train_tasks,
              valid_tasks,
              writer=None,
              
        ):
        ''' train the MPCNP
        
        Args: 
        
            train_tasks (list): list of training task data
            valid_tasks (list): list of validation task data
        '''
        
        train_losses = []
        valid_losses = []
        all_train_metrics = []
        all_valid_metrics = []
        
        num_train_tasks = len(train_tasks)
        num_valid_tasks = len(valid_tasks)
        
        for epoch in range(self.epochs):
            
            self.optimizer.zero_grad()
            
            train_target_x, train_context_x, train_target_y, train_context_y, train_target_smiles, train_context_smiles = self.generator(num_train_tasks, train_tasks)

            mu, sigma, dist = self.predict(train_context_x, train_context_y, train_target_x)
            
            # compute loss
            loss = self.loss_fn(dist, train_target_y)
            
            
            # make predictions on the validation set
            if epoch % self.pred_int == 0:
                
                # EVALUATE PREDICTION ON TRAIN TASKS -----------------------------
                
                train_r2, train_mae, train_rmse, train_pearson, train_spearman = self.evaluate_prediction(train_target_y, mu)
                
                all_train_metrics.append([train_r2, train_mae, train_rmse, train_pearson, train_spearman])
                
                # MAKE PREDICTION ON VALIDATION TASKS -------------------------
                
                valid_target_x, valid_context_x, valid_target_y, valid_context_y, valid_target_smiles, valid_context_smiles = self.generator(num_valid_tasks, valid_tasks)

                mu, sigma, dist = self.predict(valid_context_x, valid_context_y, valid_target_x)

                # compute loss
                valid_loss = self.loss_fn(dist, valid_target_y)
                
                # EVALUATE PREDICTION ON VALIDATION TASKS -----------------------------
                
                valid_r2, valid_mae, valid_rmse, valid_pearson, valid_spearman = self.evaluate_prediction(valid_target_y, mu)
                
                all_valid_metrics.append([valid_r2, valid_mae, valid_rmse, valid_pearson, valid_spearman])
                
                train_losses.append(loss)
                valid_losses.append(valid_loss)
               
                print(f'@@\tEPOCH : {epoch}\tTRAIN LOSS : {loss:.3f}\tVALID LOSS : {valid_loss:.3f}\tTRAIN R2 : {train_r2:.3f}\tVALID R2 : {valid_r2:.3f}')
                
                if writer is not None: 
                    writer.add_scalar("Loss/Train", loss, epoch)
                    writer.add_scalar("Loss/Valid", valid_loss, epoch)
                    
                    writer.add_scalar("R2/Train", train_r2, epoch)
                    writer.add_scalar("R2/Valid", valid_r2, epoch)
                    
                    writer.add_scalar("MAE/Train", train_mae, epoch)
                    writer.add_scalar("MAE/Valid", valid_mae, epoch)
                    
                    writer.add_scalar("RMSE/Train", train_rmse, epoch)
                    writer.add_scalar("RMSE/Valid", valid_rmse, epoch)
                    
                    writer.add_scalar("Pearson/Train", train_pearson, epoch)
                    writer.add_scalar("Pearson/Valid", valid_pearson, epoch)
                    
                    writer.add_scalar("Spearman/Train", train_spearman, epoch)
                    writer.add_scalar("Spearman/Valid", valid_spearman, epoch)
                
            loss.backward()
            self.optimizer.step()
            
        
        training_results = {'training': all_train_metrics, 'validation': all_valid_metrics}
        
        with open('training_results.pkl', 'wb') as f:
            pickle.dump(training_results, f)

        return None
        
        
        
    def predict(self,
                context_x, 
                context_y,
                target_x
            ): 
        ''' Forward pass of the MPCNP
        '''
        
        # representation from deterministic encoder
        rep = self.encoder.forward(context_x, context_y)

        # run decoder
        mu, sigma, dist = self.decoder.forward(target_x, rep)
    
        return mu, sigma, dist
    
    def evaluate_prediction(self, target_y, mu):
        
        metrics = []
        for true, pred in zip(target_y, mu):
            # compute metrics on train set
            vms_batch  = self.metrics.compute_metrics(true.cpu().data.numpy(), pred.cpu().data.numpy(),['r2', 'mae', 'rmse', 'pearson', 'spearman',])
            metrics.append(vms_batch)

        r2  = np.mean([t['r2'] for t in metrics])
        mae = np.mean([t['mae'] for t in metrics])
        rmse = np.mean([t['rmse'] for t in metrics]) 
        pearson = np.mean([t['pearson'] for t in metrics])
        spearman = np.mean([t['spearman'] for t in metrics]) 

        return r2, mae, rmse, pearson, spearman
        

    
