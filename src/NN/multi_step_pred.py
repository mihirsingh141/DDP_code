import os
import argparse
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
# import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from torchdiffeq import odeint
from Data.dataset import * 
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

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

class MLP(nn.Module):
    """Multi-layer perceptron used as the Hamiltonian function approximator"""
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='relu'):
        super(MLP, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim, bias=None)
        
        # Initialize weights for better training
        for l in [self.linear1, self.linear2, self.linear3]:
            nn.init.orthogonal_(l.weight)
            if hasattr(l, 'bias') and l.bias is not None:
                nn.init.constant_(l.bias, 0.0)
        
        # Set nonlinearity
        if nonlinearity == 'relu':
            self.nonlinearity = F.relu
        elif nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        else:
            self.nonlinearity = F.relu
    
    def forward(self, x):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)



class PredModel_Time(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_dim, nonlinearity='relu', num_layers=2):
        super(PredModel_Time, self).__init__() 

        # Time embedding network
        self.time_net = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.Tanh()
        )

        # Convert nonlinearity string to module
        if nonlinearity == 'relu':
            nl_layer = nn.ReLU()
        elif nonlinearity == 'tanh':
            nl_layer = nn.Tanh()
        else:
            nl_layer = nn.ReLU()  # Default

        # Build model layers
        layers = []
        layers.append(nn.Linear(input_dim+time_dim, hidden_dim))
        layers.append(nl_layer)

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            # Need to create a new instance each time
            if nonlinearity == 'relu':
                layers.append(nn.ReLU())
            elif nonlinearity == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, input_dim))

        self.model = nn.Sequential(*layers)

    def forward(self,x,t):
        batch_size = x.shape[0] 

        if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.dim() == 0):
            t = torch.ones(batch_size, 1) * t 
        
        elif isinstance(t, torch.Tensor) and t.dim() == 1:
            t = t.view(-1, 1) 


        # Embed the time
        t_embed = self.time_net(t) 

        # Concatenate time embedding with state
        combined = torch.cat([t_embed,x],dim=1) 

        return self.model(combined)
    
class PredModel(nn.Module):
    def __init__(self, n_dim, input_dim, hidden_dim, output_dim,nonlinearity='relu', num_layers=2, dropout_rate=0.2, use_batch_norm=True):
        super(PredModel, self).__init__() 

        self.n_dim = n_dim 

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
        batch_size = x.shape[0] 
        x = x.view(batch_size,-1) 

        y = self.model(x) 

        return y.reshape(batch_size,-1,self.n_dim)
    
def validate(model, test_loader,device):
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
    

def train_PredModel(train_dataset,test_dataset,batch_size,epochs,lr,n_dim,input_dim,ouptut_dim,hidden_dim=128,nonlinearity='relu',num_layers=2,debug=False,device=None):
    device = get_device() 
    model = PredModel(n_dim,input_dim,hidden_dim,ouptut_dim,nonlinearity,num_layers).to(device) 
    optimizer = optim.Adam(model.parameters(),lr=lr) 


    train_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_loader = DataLoader(test_dataset,batch_size=batch_size) 

    stats = {
        'train_loss': [], 
        'test_loss': [], 
        'grad_norm': [],
    } 

    
    best_test_loss = float('inf')
    best_model = None

    
    print(f"Starting training: {epochs} epochs")

    for epoch in tqdm(range(epochs)):
        model.train() 

        train_loss = 0
        epoch_grad_norm = 0
        num_batches = 0

        for  x,y in train_loader:
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


def create_sequences(data, input_length, target_length=1):
    """
    Convert time series data to sequences of arbitrary length
    
    Args:
        data: Tensor of shape (num_trajectories, sequence_length, state_dimension)
        input_length: Length of input sequence (t-n, t-n+1, ..., t-1)
        target_length: Length of target sequence (t, t+1, ..., t+target_length-1)
                       Default is 1 (just predict the next state)
    
    Returns:
        inputs: Input sequences of shape (num_sequences, input_length, state_dimension)
        targets: Target sequences of shape (num_sequences, target_length, state_dimension)
    """
    # Convert to tensor if it's not already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    
    num_trajectories, seq_length, state_dim = data.shape
    
    # Check if the sequence lengths are valid
    total_length = input_length + target_length
    if total_length > seq_length:
        raise ValueError(f"Total sequence length {total_length} exceeds data sequence length {seq_length}")
    
    # Calculate number of sequences per trajectory
    seqs_per_traj = seq_length - total_length + 1
    
    # Total number of sequences
    total_seqs = num_trajectories * seqs_per_traj
    
    # Initialize tensors
    inputs = torch.zeros((total_seqs, input_length, state_dim))
    targets = torch.zeros((total_seqs, target_length, state_dim))
    
    # Create the sequences
    seq_idx = 0
    for traj in range(num_trajectories):
        for t in range(seqs_per_traj):
            # Input sequence: (t, t+1, ..., t+input_length-1)
            inputs[seq_idx] = data[traj, t:t+input_length]
            
            # Target sequence: (t+input_length, ..., t+input_length+target_length-1)
            targets[seq_idx] = data[traj, t+input_length:t+input_length+target_length]
            
            seq_idx += 1
    
    return inputs, targets

if __name__ == '__main__':
    with open('data/spring_mass_dataset.pkl', 'rb') as file:
        data_dict = pickle.load(file)   

    coords = data_dict['coords']
    test_coords = data_dict['test_coords']

    X,y = create_sequences(coords,2) 
    test_X, test_y = create_sequences(test_coords,2) 


def forecast_nn(model,last_sequence,n_steps,feature_dim,device=None):
    if device is None:
        device = get_device() 

    model = model.to(device) 
    model.eval() 

    with torch.no_grad():
        sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(device) 
        sample_pred = model(sequence) 

        if len(sample_pred.shape) == 3:
            forecast_horizon = sample_pred.shape[1] 
        else:
            forecast_horizon = 1
            sample_pred = sample_pred.unsqueeze(1) 
        
        forecast = np.zeros((n_steps,feature_dim)) 

        current_sequence = sequence.clone() 

        for i in range(0,n_steps,forecast_horizon):
            next_steps = model(current_sequence) 

            if next_steps.ndim == 2:
                next_steps = next_steps.view(-1,forecast_horizon,feature_dim) 
            
            next_steps = next_steps.cpu().numpy()[0] 

            steps_to_add = min(forecast_horizon,n_steps-i) 
            forecast[i:i+steps_to_add] = next_steps[:steps_to_add]

            if i+forecast_horizon < n_steps:
                new_sequence = torch.cat([
                    current_sequence[:,forecast_horizon:,:],
                    torch.FloatTensor(next_steps).unsqueeze(0).to(device)
                ],dim=1) 
                current_sequence = new_sequence

        
    return forecast


def forecast_nn_laligan(model, autoencoder, last_sequence, n_steps, feature_dim, latent_dim,device=None):
    if device is None:
        device = get_device()  # You should define this function to choose 'cuda' or 'cpu'

    model = model.to(device)
    autoencoder = autoencoder.to(device)
    model.eval()
    autoencoder.eval()

    with torch.no_grad():
        # Convert input sequence to tensor and encode
        sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        current_sequence_enc = autoencoder.encode(sequence)

        # Get forecast horizon from one prediction
        sample_pred = model(current_sequence_enc)
        if sample_pred.ndim == 2:
            forecast_horizon = 1
            sample_pred = sample_pred.unsqueeze(1)
        else:
            forecast_horizon = sample_pred.shape[1]

        # Prepare forecast output array
        forecast = np.zeros((n_steps, feature_dim))

        for i in range(0, n_steps, forecast_horizon):
            # Predict next latent steps
            next_steps_enc = model(current_sequence_enc)

            if next_steps_enc.ndim == 2:
                next_steps_enc = next_steps_enc.view(-1, forecast_horizon, next_steps_enc.shape[-1])

            # Decode to observation space
            next_steps = autoencoder.decode(next_steps_enc)
            next_steps_np = next_steps.cpu().numpy()[0]

            # Add to forecast
            steps_to_add = min(forecast_horizon, n_steps - i)
            forecast[i:i + steps_to_add] = next_steps_np[:steps_to_add]

            if i + forecast_horizon < n_steps:
                # Re-encode the decoded steps to get latent for next iteration
                next_steps_tensor = torch.FloatTensor(next_steps_np[:steps_to_add]).unsqueeze(0).to(device)
                next_steps_enc = autoencoder.encode(next_steps_tensor)

                # Shift and update latent sequence
                current_sequence_enc = torch.cat([
                    current_sequence_enc[:, forecast_horizon:, :],
                    next_steps_enc
                ], dim=1)

    return forecast

