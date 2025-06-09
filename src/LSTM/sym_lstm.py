import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pickle
import numpy as np
import torch
import jax
import jax.numpy as jnp
import objax
from emlp.groups import Group, SO
from emlp.reps import V, T, Scalar, Vector
from emlp.nn import EMLP, Linear
from src.gan import LieGenerator
from typing import Tuple, List, Optional, Union, Callable
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

# Configure JAX to use GPU if available
if torch.cuda.is_available():
    jax.config.update('jax_platform_name', 'gpu')

device=get_device()

class CustomGroup(Group):
    def __init__(self,n,generators):
        if len(generators.shape) == 2:
            generators = np.expand_dims(generators,axis=0) 
        self.lie_algebra = generators 
        super().__init__(n)

def get_generators(n_dim,n_channel,checkpoint_gen,task):
    device = get_device()
    with open(f'saved_model/args_{task}.pkl','rb') as f:
        args = pickle.load(f)

    generator = LieGenerator(n_dim,n_channel,args)
    checkpoint_gen = torch.load(checkpoint_gen, map_location=device)
    generator.load_state_dict(checkpoint_gen)
    generator = generator.to(device)

    return np.stack([x.detach().cpu().numpy() for x in generator.getLi()],axis=1).squeeze()


class LSTMForecaster(nn.Module):
    def __init__(self, n_dim, hidden_dim, fc_dim, output_dim, num_layers, dropout=0.0):
        """
        LSTM model for time series forecasting with two FC layers
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Number of hidden units in LSTM
        fc_dim : int
            Number of hidden units in the first FC layer
        num_layers : int
            Number of LSTM layers
        output_dim : int
            Number of output features
        dropout : float
            Dropout probability (applied between layers)
        """
        super(LSTMForecaster, self).__init__()
        
        self.n_dim = n_dim 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
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
        self.fc2 = nn.Linear(fc_dim, output_dim*n_dim) 

    def forward(self,x):
        bs = x.shape[0]
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (batch_size, seq_length, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        
        out = out[:, -1, :] # last timestep

        out = self.fc1(out) 
        out = self.relu(out) 
        out = self.dropout(out)

        out = self.fc2(out) 
        out = out.reshape(bs,-1,self.n_dim)

        return out


# def train_LSTMModel(train_dataset, test_dataset, batch_size, epochs, lr, n_dim, output_dim,
#                    hidden_dim=128, fc_dim=64, num_layers=2, dropout=0.2, debug=False):
#     # Initialize model
#     model = LSTMForecaster(n_dim, hidden_dim, fc_dim,output_dim, num_layers, dropout).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     # Create data loaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
#     # Initialize statistics dictionary
#     stats = {
#         'train_loss': [],
#         'test_loss': [],
#         'grad_norm': [],
#         'batch_train_loss': [],
#         'batch_test_loss': []
#     }
    
#     # Track best model for early stopping
#     best_test_loss = float('inf')
#     best_model = None
    
#     print(f"Starting training: {epochs} epochs")
    
#     for epoch in tqdm(range(epochs)):
#         model.train()
        
#         train_loss = 0
#         num_batches = 0
        
#         for x, y in train_loader:
#             if x is None or y is None:
#                 print('Warning: Got None for batch')
#                 continue 

#             x=x.to(device)
#             y = y.to(device)
#             try:
#                 # Forward pass
#                 pred = model(x)
#                 loss = F.mse_loss(pred, y)
                
#                 # Backward pass
#                 optimizer.zero_grad()
#                 loss.backward()
                
#                 # Calculate gradient norm
#                 grad_norm = 0.0
#                 for p in model.parameters():
#                     if p.grad is not None:
#                         grad_norm += p.grad.norm().item() ** 2
#                 grad_norm = grad_norm ** 0.5
                
#                 # Gradient clipping
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
#                 # Update parameters
#                 optimizer.step()
                
#                 # Track metrics
#                 train_loss += loss.item()
#                 num_batches += 1
                
#                 # Calculate batch-level metrics
#                 batch_train_loss = loss.item()
#                 batch_test_loss = validate(model, test_loader)
                
#                 # Store batch-level metrics
#                 stats['batch_train_loss'].append(batch_train_loss)
#                 stats['batch_test_loss'].append(batch_test_loss)
                
#                 if debug and num_batches % 10 == 0:
#                     print(f"Batch {num_batches}, Loss: {loss.item():.6f}, Grad norm: {grad_norm:.6f}")
                    
#             except Exception as e:
#                 print(f'Error in training batch: {e}')
        
#         if num_batches == 0:
#             print("No valid batches in training epoch")
#             continue
        
#         # Calculate epoch-level metrics
#         train_loss /= num_batches
#         grad_norm /= num_batches
#         stats['train_loss'].append(train_loss)
#         stats['grad_norm'].append(grad_norm)
        
#         # Run full validation on test set
#         test_loss = validate(model, test_loader)
#         stats['test_loss'].append(test_loss)
        
#         # Print progress
#         if epoch % 10 == 0 or epoch == epochs - 1:
#             print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
#         # Keep track of best model
#         if test_loss < best_test_loss:
#             best_test_loss = test_loss
#             best_model = model.state_dict().copy()

#         # Setting the model back to train 
#         model.train()
        
#         # Early convergence check
#         if test_loss < 1e-6:
#             print(f"Early stopping at epoch {epoch+1} - test loss: {test_loss:.8f}")
#             break
    
#     # Load best model
#     if best_model is not None:
#         model.load_state_dict(best_model)
    
#     return model, stats

def train_LSTMModel(train_dataset, test_dataset, batch_size, epochs, lr, n_dim, output_dim,
                   hidden_dim=128, fc_dim=64, num_layers=2, dropout=0.2, debug=False):
    # Initialize model
    model = LSTMForecaster(n_dim, hidden_dim, fc_dim, output_dim, num_layers, dropout).to(device)
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
    
    # Separate validation function that creates a new model copy for validation
    def validate_epoch(val_model, data_loader):
        val_model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(device)
                y = y.to(device)
                try:
                    pred = val_model(x)
                    loss = F.mse_loss(pred, y)
                    
                    num_batches += 1
                    val_loss += loss.item()
                
                except Exception as e:
                    print(f'Error in validation batch: {e}')
                    continue
        
        val_model.train()
        if num_batches == 0:
            return float('inf')
        else:
            return val_loss / num_batches
    
    for epoch in tqdm(range(epochs)):
        # Ensure model is in training mode at the start of each epoch
        model.train()
        
        train_loss = 0
        num_batches = 0
        epoch_grad_norm = 0
        
        for x, y in train_loader:
            if x is None or y is None:
                print('Warning: Got None for batch')
                continue 

            x = x.to(device)
            y = y.to(device)
            
            # Double-check we're in training mode before each batch
            model.train()
            
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
                batch_loss = loss.item()
                train_loss += batch_loss
                num_batches += 1
                
                if debug and num_batches % 10 == 0:
                    print(f"Batch {num_batches}, Loss: {batch_loss:.6f}, Grad norm: {grad_norm:.6f}")
                    
            except Exception as e:
                print(f'Error in training batch: {e}')
                import traceback
                traceback.print_exc()  # Print the full error trace
                continue
        
        if num_batches == 0:
            print("No valid batches in training epoch")
            continue
        
        # Calculate epoch-level metrics
        train_loss /= num_batches
        epoch_grad_norm /= num_batches
        stats['train_loss'].append(train_loss)
        stats['grad_norm'].append(epoch_grad_norm)
        
        # Run validation using our separate function
        test_loss = validate_epoch(model, test_loader)
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

# def validate(model, test_loader):
#     """Run validation on the provided data"""
#     model.eval()
#     val_loss = 0.0
#     num_batches = 0
    
#     with torch.no_grad():
#         for x, y in test_loader:
#             x=x.to(device)
#             y=y.to(device)
#             try:
#                 pred = model(x)
#                 loss = F.mse_loss(pred, y)
                
#                 num_batches += 1
#                 val_loss += loss.item()
            
#             except Exception as e:
#                 print(f'Error in validation batch: {e}')
#                 continue
    
#     if num_batches == 0:
#         return float('inf')
#     else:
#         return val_loss / num_batches
    

def forecast_future(model,last_sequence,n_steps,feature_dim,device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model = model.to(device)
    model.eval() 

    # Convert to tensor and add batch dimension
    sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
    forecast_horizon = model(sequence).size(1) 

    forecast = np.zeros((n_steps,feature_dim)) 

    current_sequence = sequence.clone() 

    for i in range(0,n_steps,forecast_horizon):
        with torch.no_grad():
            next_steps = model(current_sequence) 

            if next_steps.ndim == 2:
                next_steps = next_steps.view(-1, forecast_horizon, feature_dim)


            next_steps = next_steps.cpu().numpy()[0] 

        # add to forecast 
        steps_to_add = min(forecast_horizon,n_steps-i) 
        forecast[i:i+steps_to_add] = next_steps[:steps_to_add]

        if i+forecast_horizon < n_steps:
            new_sequence = torch.cat([
                current_sequence[:,forecast_horizon:,:],
                torch.FloatTensor(next_steps).unsqueeze(0).to(device)
            ],dim=1) 
            current_sequence = new_sequence

        
    return forecast



class EMLP_lstm(objax.Module):
    """LSTM implementation using EMLP components for equivariant processing""" 

    def __init__(self,group,input_dim,hidden_dim,output_dim,num_layers=2):
        super().__init__() 

        self.group = group
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.input_rep = Vector * input_dim 
        self.hidden_rep = Vector * hidden_dim 
        self.cell_rep = Vector * hidden_dim 
        self.output_rep = Vector * output_dim 

        self.lstm_layers = []

        for layer in range(num_layers):
            layer_in_dim = input_dim if layer == 0 else hidden_dim
            layer_in_rep = self.input_rep if layer == 0 else self.hidden_rep

            combined_in_rep = layer_in_rep(self.group) + self.hidden_rep(self.group) 
            combined_out_rep = (self.hidden_rep * 4)(self.group) 

            gates_emlp = EMLP(
                combined_in_rep,
                combined_out_rep,
                group=self.group,
                num_layers=2,
                ch=hidden_dim
            )

            if layer == num_layers-1:
                output_emlp = EMLP(
                    self.hidden_rep(self.group),
                    self.output_rep(self.group),
                    group=self.group,
                    num_layers=1
                )
            else:
                output_emlp=None 

        self.lstm_layers.append((gates_emlp,output_emlp))

    def init_hidden(self,batch_size):
        hidden = jnp.zeros((self.num_layers, batch_size, self.hidden_dim))
        cell = jnp.zeros((self.num_layers, batch_size, self.hidden_dim))
        return hidden, cell

    def lstm_cell_forward(self,x,hidden,cell,layer_idx):
        gates_emlp, _ = self.lstm_layers[layer_idx] 
        combined_input = jnp.concatenate([x, hidden], axis=-1) 

        gates = gates_emlp(combined_input) # gates outputs

        gate_size = self.hidden_dim 
        i, f, g, o = jnp.split(gates, 4, axis=-1) 

        # Apply gate activations
        i = jax.nn.sigmoid(i)  # Input gate
        f = jax.nn.sigmoid(f)  # Forget gate
        g = jnp.tanh(g)        # Cell update
        o = jax.nn.sigmoid(o)  # Output gate
        
        # Update cell state
        new_cell = f * cell + i * g
        
        # Update hidden state
        new_hidden = o * jnp.tanh(new_cell)
        
        return new_hidden, new_cell
    
    def __call__(self,x,hidden=None,return_sequence=None):
        batch_size, seq_length, _ = x.shape
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden, cell = self.init_hidden(batch_size)
        else:
            hidden, cell = hidden

        if return_sequence:
            outputs = [] 

        for t in range(seq_length):
            x_t = x[:,t,:] 
            layer_input = x_t 

            for layer_idx in range(self.num_layers):
                layer_h = hidden[layer_idx]
                layer_c = cell[layer_idx] 

                new_h, new_c = self.lstm_cell_forward(layer_input, layer_h, layer_c, layer_idx)

                hidden = hidden.at[layer_idx].set(new_h)
                cell = cell.at[layer_idx].set(new_c) 

                layer_input = new_h 

            if return_sequence:
                if self.lstm_layers[-1][1] is not None:
                    out = self.lstm_layers[-1][1](layer_input)
                else:
                    out = layer_input
                outputs.append(out)

        if return_sequence:
            return jnp.stack(outputs,axis=1) 
        else:
            if self.lstm_layers[-1][1] is not None:
                return self.lstm_layers[-1][1](layer_input)
            else:
                return layer_input 
            

def train_emlp_lstm(group,input_dim,hidden_dim,output_dim,num_layers, train_dataset, test_dataset, num_epochs=100, lr=0.001, 
                   batch_size=32, seq_length=50, forecast_length=10):
    model = EMLP_lstm(group,input_dim,hidden_dim,output_dim,num_layers)
    opt = objax.optimizer.Adam(model.vars())

    def validate_emlp(loader):
        total_loss = 0.0
        num_batches = 0

        for x,y in loader:
            bs = x.shape[0]
            x_jax = jnp.array(x)
            y_jax = jnp.array(y).reshape(bs,-1)

            batch_loss = loss_fn(x_jax,y_jax)
            total_loss += batch_loss
            num_batches += 1

        return total_loss/num_batches

    
    @objax.Function.with_vars(model.vars())
    def loss_fn(x, y):
        bs = x.shape[0]
        pred = model(x, return_sequence=True).reshape(bs,-1)
        return jnp.mean((pred - y) ** 2)
    
    grad_and_val = objax.GradValues(loss_fn, model.vars())
    
    @objax.Function.with_vars(model.vars() + opt.vars())
    def train_op(x, y, lr):
        g, v = grad_and_val(x, y)
        opt(lr=lr, grads=g)
        return v
    
    stats = {
        'train_loss':[],
        'test_loss':[]
    }

    train_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_loader = DataLoader(test_dataset,batch_size=batch_size)

    for epoch in tqdm(range(num_epochs)):
        train_loss = 0
        num_batches = 0

        for x,y in train_loader:
            try:
                bs = x.shape[0]
                x_jax = jnp.array(x) 
                y_jax = jnp.array(y).reshape(bs,-1)

                batch_loss = train_op(x_jax,y_jax,lr)
            
                if isinstance(batch_loss,list):
                    batch_loss = batch_loss[0].mean() 
                train_loss += float(batch_loss)

                num_batches += 1

            except Exception as e: 
                print(f"Error in training batch: {e}")
                continue
                
        if num_batches == 0:
            print("No valid batches in training epoch")
            continue

        train_loss /= num_batches
        stats['train_loss'].append(train_loss) 

        test_loss = validate_emlp(test_loader)
        stats['test_loss'].append(test_loss)

        # Print progress 
        if epoch % 10 == 0 or epoch == epoch - 1:
            print(f"Epoch {epoch+1}/{epoch}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

        # Early convergence check 
        if test_loss < 1e-6:
            print(f"Early stopping at epoch {epoch+1} - test loss: {test_loss:.8f}")
            break

    
    return model, stats
    

class TrajectoryDataset(Dataset):
    def __init__(self,trajectories,input_seq_length,forecast_length):
        super().__init__()
        
        self.trajectories = trajectories 
        self.input_seq_length = input_seq_length 
        self.forecast_length = forecast_length 

        self.samples = self.create_samples()

    def create_samples(self):
        samples = []
        for traj in self.trajectories:
            # Calculate the number of possible samples from this trajectory
            num_samples = len(traj) - self.input_seq_length - self.forecast_length + 1
            
            # Create samples with sliding window
            for i in range(num_samples):
                input_seq = traj[i:i+self.input_seq_length]
                target_seq = traj[i+self.input_seq_length:i+self.input_seq_length+self.forecast_length]
                samples.append((input_seq, target_seq))
        
        return samples
    
    def __len__(self):
        return len(self.samples) 
    
    def __getitem__(self, index):
        input_seq, target_seq = self.samples[index] 

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)
        target_tensor = torch.tensor(target_seq, dtype=torch.float32)
        
        return input_tensor, target_tensor
    

