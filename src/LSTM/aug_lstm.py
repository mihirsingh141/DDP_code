import os
import argparse
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm 
from src.gan import LieGenerator 
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add device configuration
def get_device():
    """Get the appropriate device for computation"""
    if torch.cuda.is_available():
        return torch.device('cuda:3')
    return torch.device('cpu')

device = get_device()

def sampleFromLieGroup(generator, n_samples=1):
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

    generator = LieGenerator(n_dim,args.n_channel,args).to(device) 

    return generator


class AugLSTMForecaster(nn.Module):
    def __init__(self, generator, n_dim, hidden_dim, fc_dim, output_dim, num_layers, 
                 dropout=0.0, aug_eval=True, n_copy=4):
        """
        LSTM model with Lie group data augmentation for time series forecasting
        
        Parameters:
        -----------
        generator : LieGenerator
            Generator for Lie group transformations
        n_dim : int
            Dimension of each feature (e.g., 3 for x,y,z coordinates)
        hidden_dim : int
            Number of hidden units in LSTM
        fc_dim : int
            Number of hidden units in the first FC layer
        output_dim : int
            Number of output time steps
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout probability
        aug_eval : bool
            Whether to use augmentation during evaluation
        n_copy : int
            Number of augmentation copies during evaluation
        """
        super(AugLSTMForecaster, self).__init__()
        
        self.generator = generator
        self.n_dim = n_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.aug_eval = aug_eval
        self.n_copy = n_copy
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # First fully connected layer
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Second fully connected layer (output layer)
        self.fc2 = nn.Linear(fc_dim, output_dim * n_dim)
    
    def base_forward(self, x):
        """The core LSTM forward pass without augmentation"""
        batch_size = x.shape[0]
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Use only the last time step's output
        out = out[:, -1, :]
        
        # Pass through the fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Reshape to (batch_size, output_time_steps, n_dim)
        out = out.reshape(batch_size, self.output_dim, self.n_dim)
        
        return out

    def forward(self, x):
        batch_size = x.shape[0]
        
        if self.training:
            # Apply random augmentation during training
            g, g_inv = sampleFromLieGroup(self.generator, batch_size)
            g, g_inv = g.to(x.device), g_inv.to(x.device)
            
            # Transform input sequence - apply same transformation to all time steps
            # x shape: (batch_size, seq_length, n_dim)
            # g shape: (batch_size, n_dim, n_dim)
            x_transformed = torch.einsum('bjk,btk->btj', g, x)
            
            # Process through network
            out = self.base_forward(x_transformed)
            
            # Inverse transform the output
            y = torch.einsum('bjk,btk->btj', g_inv, out)
            
        elif self.aug_eval and self.n_copy > 1:
            # Use multiple augmentations during evaluation
            outputs = []
            
            for _ in range(self.n_copy):
                # Generate a different transformation for each copy
                g, g_inv = sampleFromLieGroup(self.generator, batch_size)
                g, g_inv = g.to(x.device), g_inv.to(x.device)
                
                # Transform input
                x_transformed = torch.einsum('bjk,btk->btj', g, x)
                
                # Process through network
                out = self.base_forward(x_transformed)
                
                # Inverse transform
                y_aug = torch.einsum('bjk,btk->btj', g_inv, out)
                outputs.append(y_aug)
            
            # Average the outputs for final prediction
            y = torch.mean(torch.stack(outputs), dim=0)
            
        else:
            # Regular forward pass without augmentation
            y = self.base_forward(x)
        
        return y

def train_AugLSTMModel(group,train_dataset, test_dataset, batch_size, epochs, lr, n_dim, output_dim,
                   hidden_dim=128, fc_dim=64, num_layers=2, dropout=0.2, debug=False):
    # Initialize model
    model = AugLSTMForecaster(group,n_dim, hidden_dim, fc_dim, output_dim, num_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize statistics dictionary
    stats = {
        'train_loss': [],
        'test_loss': [],
        'grad_norm': [],
    }
    
    # Track best model for early stopping
    best_test_loss = float('inf')
    best_model = None
    
    print(f"Starting training: {epochs} epochs")
    
    for epoch in tqdm(range(epochs)):
        model.train()
        
        train_loss = 0
        epoch_grad_norm = 0
        num_batches = 0
        
        for x, y in train_loader:
            x = x.to(device) 
            y = y.to(device)

            try:
                # Forward pass
                pred = model(x)
                loss = F.mse_loss(pred, y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Calculate gradient norm
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                epoch_grad_norm += grad_norm
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                num_batches += 1
                
                
                if debug and num_batches % 10 == 0:
                    print(f"Batch {num_batches}, Loss: {loss.item():.6f}, Grad norm: {grad_norm:.6f}")
                    
            except Exception as e:
                print(f'Error in training batch: {e}')
        
        if num_batches == 0:
            print("No valid batches in training epoch")
            continue
        
        # Calculate epoch-level metrics
        train_loss /= num_batches
        epoch_grad_norm /= num_batches
        stats['train_loss'].append(train_loss)
        stats['grad_norm'].append(grad_norm)
        
        # Run full validation on test set
        test_loss = validate(model, test_loader)
        stats['test_loss'].append(test_loss)
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        # Keep track of best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = model.state_dict().copy()
        
        # Early convergence check
        if test_loss < 1e-6:
            print(f"Early stopping at epoch {epoch+1} - test loss: {test_loss:.8f}")
            break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model, stats


def validate(model, test_loader):
    """Run validation on the provided data"""
    model.eval()
    val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            try:
                pred = model(x)
                loss = F.mse_loss(pred, y)
                
                num_batches += 1
                val_loss += loss.item()
            
            except Exception as e:
                print(f'Error in validation batch: {e}')
                continue
    
    model.train()
    if num_batches == 0:
        return float('inf')
    else:
        return val_loss / num_batches


