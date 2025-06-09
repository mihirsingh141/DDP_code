import os
import argparse
import time
import pickle
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm 
from src.gan import LieGenerator 
from src.NN.multi_step_pred import PredModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Add device configuration
def get_device():
    """Get the appropriate device for computation"""
    if torch.cuda.is_available():
        return torch.device('cuda:3')
    return torch.device('cpu')


def sampleFromLieGroup(generator, n_samples=1, device=None):
    if device == None:
        device = get_device() 

    generator = generator.to(device)
    z = generator.sample_coefficient(n_samples, device=device)
    if z.ndim == 1:
        z = z.unsqueeze(0)  # ensure shape (n_samples, c)
        
    Li = generator.getLi()  # shape (c, j, k)
    
    g = torch.matrix_exp(torch.einsum('bc,cjk->bjk', z, Li))
    g_inv = torch.matrix_exp(torch.einsum('bc,cjk->bjk', -z, Li))

    return g, g_inv


def get_generator(n_dim,args_path):
    with open(args_path,'rb') as f:
        args = pickle.load(f) 

    generator = LieGenerator(n_dim,args.n_channel,args).to(args.device) 

    return generator

class Aug_PredModel(nn.Module):
    def __init__(self, generator, n_dim, input_dim, hidden_dim, output_dim, nonlinearity='relu', num_layers=2, aug_eval=True, n_copy=4, dropout_rate=0.2, use_batch_norm=True):
        super(Aug_PredModel, self).__init__()  

        self.generator = generator
        self.aug_eval = aug_eval 
        self.n_copy = n_copy
        self.n_dim = n_dim
        self.input_dim = input_dim 
        self.output_dim = output_dim

        # Convert nonlinearity string to module
        if nonlinearity == 'relu':
            nl_layer = nn.ReLU()
        elif nonlinearity == 'tanh':
            nl_layer = nn.Tanh()
        else:
            nl_layer = nn.Sigmoid()  # Default

        # Build model layers
        layers = []
        layers.append(nn.Linear(input_dim*n_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nl_layer)
        layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nl_layer)
            layers.append(nn.Dropout(dropout_rate))
            

        layers.append(nn.Linear(hidden_dim, output_dim*n_dim))

        self.model = nn.Sequential(*layers)


    def forward(self,x):
        b = x.shape[0]  # Batch size
    
        
        if self.training:
            g, g_inv = sampleFromLieGroup(self.generator, b)
            g, g_inv = g.to(x.device), g_inv.to(x.device)
            gx = torch.einsum('bjk,btk->btj', g, x).reshape(b, -1)
            gy = self.model(gx).view(b, self.output_dim, -1)
            y = torch.einsum('bjk, btk->btj', g_inv, gy)
        elif self.aug_eval:
            x_copies = []
            g_inv_list = []
            for _ in range(self.n_copy):
                g, g_inv = sampleFromLieGroup(self.generator, b)
                g, g_inv = g.to(x.device), g_inv.to(x.device)
                gx = torch.einsum('bjk,btk->btj', g, x).reshape(b, -1)
                x_copies.append(gx)
                g_inv_list.append(g_inv)
            x_aug = torch.cat(x_copies, dim=0)  # (N*b, tin*k)
            g_inv = torch.cat(g_inv_list, dim=0)  # (N*b, k, k)
            y_aug = self.model(x_aug).view(self.n_copy * b, self.output_dim, -1)  # (N*b, tout, k)
            y_aug = torch.einsum('bjk, btk->btj', g_inv, y_aug)
            y = torch.mean(torch.stack(torch.split(y_aug, b)), dim=0)
        else:
            y = self.model(x.view(b, -1))
        return y.reshape(b, -1)
    
    
def validate(model, test_loader,device):
    """Run validation on the provided data"""
    model.eval()
    val_loss = 0.0
    num_batches = 0  

    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device) 
            
            b = x.shape[0]
            y = y.reshape(b,-1)
            try:
                pred = model(x) 
                loss = F.mse_loss(pred,y) 

                num_batches += 1
                val_loss += loss.item()
            
            except Exception as e:
                print(f'Error in validation batch: {e}') 
                continue 

    model.train()
    if num_batches == 0:
        return float('inf') 
    else:
        return val_loss/num_batches 


def train_augPred(generator,train_dataset,test_dataset,batch_size,epochs,lr,n_dim, input_dim,output_dim,hidden_dim=128,nonlinearity='relu',num_layers=2,aug_eval=True,n_copy=4,debug=False,device=None): 
    if device == None:
        device = get_device() 

    model = Aug_PredModel(generator, n_dim, input_dim, hidden_dim, output_dim, nonlinearity, num_layers, aug_eval, n_copy).to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr) 


    train_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_loader = DataLoader(test_dataset,batch_size=batch_size) 

    stats = {
        'train_loss': [], 
        'test_loss': [], 
        'grad_norm': [],
        'batch_train_loss': [],
        'batch_test_loss': []
    } 

    
    best_test_loss = float('inf')
    best_model = None

    print(f'Starting training: {epochs} epochs')
    for epoch in tqdm(range(epochs)):
        model.train() 

        train_loss = 0
        epoch_grad_norm = 0
        num_batches = 0

        for  i,(x,y) in enumerate(train_loader):
            x = x.to(device) 
            y = y.to(device) 

            b = x.shape[0]
            y = y.reshape(b,-1)

            try:
                pred = model(x) 
                loss = F.mse_loss(pred,y) 

                optimizer.zero_grad() 
                loss.backward()

                grad_norm = 0.0 
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                epoch_grad_norm += grad_norm

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 


                train_loss += loss.item() 
                num_batches += 1


                optimizer.step()

            except Exception as e:
                print(f'Error in training batch: {e}')

            if debug and num_batches % 10 == 0:
                print(f"Batch {num_batches}, Loss: {loss.item():.6f}, Grad norm: {grad_norm:.6f}") 

        if num_batches == 0:
            print("No valid batches in training epoch")
            continue


        # Calculate epoch-level metrics
        train_loss /= num_batches
        epoch_grad_norm /= num_batches
        stats['train_loss'].append(train_loss)
        stats['grad_norm'].append(epoch_grad_norm)
        

        # Run full validation on test set
        test_loss = validate(model, test_loader,device)
        stats['test_loss'].append(test_loss) 


        # Print progress 
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss 
            best_model = model.state_dict().copy() 

        # Early convergence check 
        if test_loss < 1e-6:
            print(f"Early stopping at epoch {epoch+1} - test loss: {test_loss:.8f}")
            break

    if best_model is not None:
        model.load_state_dict(best_model) 

    return model, stats

def aug_util(dataset, generator, n_copy, device, batch_size=32):
    x_aug = []
    y_aug = []
    
    for (x, y) in DataLoader(dataset, batch_size=batch_size):
        x = x.to(device)
        y = y.to(device)

        for _ in range(n_copy):
            g, _ = sampleFromLieGroup(generator)
            g = g.to(x.device)

            gx = torch.einsum('bjk,btk->btj', g, x).detach()
            gy = torch.einsum('bjk,btk->btj', g, y).detach()

            x_aug.append(gx)
            y_aug.append(gy)

    # Stack into single tensors
    x_aug = torch.cat(x_aug, dim=0)
    y_aug = torch.cat(y_aug, dim=0)

    return x_aug, y_aug

def validate_vanilla(model, test_loader,device):
    """Run validation on the provided data"""
    model.eval()
    val_loss = 0.0
    num_batches = 0  

    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device) 
            
          
            try:
                pred = model(x) 
                loss = F.mse_loss(pred,y) 

                num_batches += 1
                val_loss += loss.item()
            
            except Exception as e:
                print(f'Error in validation batch: {e}') 
                continue 

    model.train()
    if num_batches == 0:
        return float('inf') 
    else:
        return val_loss/num_batches 


        

def train_augPred_vanilla(generator,train_dataset,test_dataset,batch_size,epochs,lr,n_dim, input_dim,output_dim,hidden_dim=128,nonlinearity='relu',num_layers=2,n_copy=4,debug=False,dropout=0.2,batch_norm=True,device=None):
    if device == None: 
        device = get_device() 

    model = PredModel(n_dim, input_dim, hidden_dim, output_dim,nonlinearity, num_layers,dropout,batch_norm).to(device) 
    optimizer = optim.Adam(model.parameters(),lr=lr) 

    x_train, y_train = aug_util(train_dataset, generator, n_copy, device, batch_size)
    x_test, y_test = aug_util(test_dataset, generator, n_copy, device, batch_size)

    train_dataset_aug = TensorDataset(x_train,y_train) 
    test_dataset_aug = TensorDataset(x_test,y_test)

    train_loader = DataLoader(train_dataset_aug,batch_size=batch_size)
    test_loader = DataLoader(test_dataset_aug,batch_size=batch_size) 

    stats = {
        'train_loss': [], 
        'test_loss': [], 
        'grad_norm': [],
    } 

    
    best_test_loss = float('inf')
    best_model = None

    print(f'Starting training: {epochs} epochs')
    for epoch in tqdm(range(epochs)):
        model.train() 

        train_loss = 0
        epoch_grad_norm = 0
        num_batches = 0

        for  i,(x,y) in enumerate(train_loader):
            x = x.to(device) 
            y = y.to(device) 

           
            try:
                pred = model(x) 
                loss = F.mse_loss(pred,y) 

                optimizer.zero_grad() 
                loss.backward()

                grad_norm = 0.0 
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                epoch_grad_norm += grad_norm 


                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 


                train_loss += loss.item() 
                num_batches += 1

                optimizer.step()

            except Exception as e:
                print(f'Error in training batch: {e}')

            if debug and num_batches % 10 == 0:
                print(f"Batch {num_batches}, Loss: {loss.item():.6f}, Grad norm: {grad_norm:.6f}") 

        if num_batches == 0:
            print("No valid batches in training epoch")
            continue


        # Calculate epoch-level metrics
        train_loss /= num_batches
        epoch_grad_norm /= num_batches
        stats['train_loss'].append(train_loss)
        stats['grad_norm'].append(epoch_grad_norm)
        

        # Run full validation on test set
        test_loss = validate_vanilla(model, test_loader,device)
        stats['test_loss'].append(test_loss) 


        # Print progress 
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss 
            best_model = model.state_dict().copy() 

        # Early convergence check 
        if test_loss < 1e-6:
            print(f"Early stopping at epoch {epoch+1} - test loss: {test_loss:.8f}")
            break

    if best_model is not None:
        model.load_state_dict(best_model) 

    return model, stats



    




