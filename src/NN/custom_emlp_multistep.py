import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import emlp
from emlp.groups import Group, S, Z, SO
from emlp.reps import V,T,vis, Scalar, Vector
from emlp.nn import EMLP
from src.gan import LieGenerator
from scipy.integrate import solve_ivp
import objax 
import jax.numpy as jnp 
from jax import vmap, grad
import jax

# At the beginning of your script
np.random.seed(42)
if hasattr(torch, 'manual_seed'):
    torch.manual_seed(42)
if hasattr(jax.random, 'PRNGKey'):
    key = jax.random.PRNGKey(42)

def get_device():
    """Get the appropriate device for computation"""
    if torch.cuda.is_available():
        return torch.device('cuda:3')
    return torch.device('cpu')

# Configure JAX to use GPU if available
if torch.cuda.is_available():
    jax.config.update('jax_platform_name', 'gpu')

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

# class PredModel_EMLP(objax.Module):
#     def __init__(self, group, n_dim, input_dim, hidden_dim, output_dim,num_layers=3,dropout_rate=0.2,use_batch_norm=True):
#         super().__init__()
#         self.G = group
#         self.n_dim = n_dim
        
#         # Define the EMLP architecture
#         # rep_in = Vector(self.G)
#         # rep_out = T(0)(self.G)  # Scalar output for Hamiltonian 

#         rep_in = Vector*input_dim 
#         rep_out = Vector*output_dim 
         
    
#         self.pred_model = EMLP(
#             rep_in(self.G),
#             rep_out(self.G),
#             group=self.G,
#             num_layers=num_layers,
#             ch=hidden_dim,
#             # dropout_rate=dropout_rate if dropout_rate >0 else None, 
#             # bn=use_batch_norm
#         )

    

#     def __call__(self,x):
#         if x.ndim > 2:
#             batch_size = x.shape[0]
#             timesteps = x.shape[1]
            
#             # Reshape to (batch_size * timesteps, n_dim)
#             x_flat = x.reshape(-1, self.n_dim)
            
#             # Process through EMLP
#             out = self.pred_model(x_flat)
                
#             # Reshape back to match input format
#             return out.reshape(batch_size, timesteps, self.n_dim)
#         else:
#             # Single sample case
#             return self.pred_model(x)


class PredModel_EMLP(objax.Module):
    def __init__(self, group, n_dim, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.G = group
        self.n_dim = n_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        

        from emlp.reps import Vector  # ensure this is imported correctly
        rep_in = Vector * input_dim
        rep_out = Vector * output_dim

        self.pred_model = EMLP(
            rep_in(self.G),
            rep_out(self.G),
            group=self.G,
            num_layers=num_layers,
            ch=hidden_dim,
        )

   

    def __call__(self, x):
        
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        # Check if flattened shape matches EMLP expectation
        expected_input_dim = self.input_dim * self.n_dim
        if x_flat.shape[1] != expected_input_dim:
            raise ValueError(f"Expected input size {expected_input_dim}, got {x_flat.shape[1]}")

        out = self.pred_model(x_flat)
        # out = out.reshape(batch_size, self.output_dim, self.n_dim)
        return out


def train_predModel_emlp(G,train_dataset,test_dataset,batch_size,epochs,lr,n_dim,input_dim,output_dim,hidden_dim=128,num_layers=3,debug=False):
    model = PredModel_EMLP(G,n_dim,input_dim,hidden_dim,output_dim,num_layers) 
    opt = objax.optimizer.Adam(model.vars())

    train_loader = DataLoader(train_dataset,batch_size)
    test_loader = DataLoader(test_dataset,batch_size)


    @objax.Jit
    @objax.Function.with_vars(model.vars())
    def loss(x,y):
        y_hat = model(x)
        return jnp.mean((y_hat - y) ** 2)
    
    grad_and_val = objax.GradValues(loss,model.vars()) 

    @objax.Jit
    @objax.Function.with_vars(model.vars()+opt.vars())
    def train_op(x,y,lr):
        g,v = grad_and_val(x,y) # returns the gradients and loss values 
        opt(lr=lr,grads=g)
        return g,v 
    
    
    def evaluate_model(loader):
        total_loss = 0.0
        num_batches = 0

        for x, y in loader:
            bs = x.shape[0]
            x = x.reshape(bs,-1)
            y = y.reshape(bs,-1)
            batch_loss = loss(jnp.array(x), jnp.array(y))
            total_loss += batch_loss
            num_batches += 1

        return total_loss / num_batches 
    

    print(f"Starting training: {epochs} epochs")

    stats = {
        'train_loss': [], 
        'test_loss': [], 
        'grad_norm': [],
    } 


    for epoch in tqdm(range(epochs)):
        # Training 
        train_loss = 0.0
        train_grad_norm = 0.0
        num_batches = 0

        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_size = batch_x.shape[0]
            batch_x = batch_x.reshape(batch_size,-1)
            batch_y = batch_y.reshape(batch_size,-1)

            if debug:
                print(f'Batch {i}:')
                print(f'Batch X shape: {batch_x.shape}') 
                print(f'Batch y shape: {batch_y.shape}')
                
            try:
                # Convert to JAX arrays
                batch_x_jax = jnp.array(batch_x)
                batch_y_jax = jnp.array(batch_y)
                
                # Train on this batch
                g_norm, batch_loss_val = train_op(batch_x_jax, batch_y_jax,lr)

                
                if debug:
                    print(f'Loss: {batch_loss_val}')
                    print(f'Grad norm: {g_norm}')
                    
                if isinstance(batch_loss_val, list):
                    batch_loss_val = batch_loss_val[0].mean()
                train_loss += float(batch_loss_val)

                if isinstance(g_norm, list):
                    g_norm = g_norm[0].mean()
                train_grad_norm += float(g_norm)
                
                
                # Update overall counters
                num_batches += 1

            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
                
        if num_batches == 0:
            print("No valid batches in training epoch")
            continue

        # Calculate epoch-level metrics
        train_loss /= num_batches
        train_grad_norm /= num_batches
        stats['train_loss'].append(train_loss)
        stats['grad_norm'].append(train_grad_norm)
        
        # Run full validation on test set
        test_loss = evaluate_model(test_loader)
        stats['test_loss'].append(test_loss)

        # Print progress 
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

        # Early convergence check 
        if test_loss < 1e-6:
            print(f"Early stopping at epoch {epoch+1} - test loss: {test_loss:.8f}")
            break

    

    return model, stats
    



def forecast_emlp(model, last_sequence, n_steps, feature_dim):
    import jax
    import jax.numpy as jnp
    import numpy as np

    # Select device
    devices = jax.devices("gpu")
    device = devices[0] if devices else jax.devices("cpu")[0]

    # Prepare initial sequence (shape: [1, T, D])
    sequence = jnp.array(last_sequence)[None, ...]
    sequence = jax.device_put(sequence, device)

    # JIT-compile model
    model_jit = jax.jit(model)

    # Run once to get forecast horizon
    sample_pred = model_jit(sequence).reshape(1,-1,feature_dim)
    forecast_horizon = sample_pred.shape[1]

    # Init forecast buffer
    forecast = np.zeros((n_steps, feature_dim))

    current_sequence = sequence

    for i in range(0, n_steps, forecast_horizon):
        # Predict next steps
        next_steps = model_jit(current_sequence).reshape(-1,forecast_horizon,feature_dim)

        if next_steps.ndim == 2:  # [1, D]
            next_steps = next_steps[:, None, :]  # [1, 1, D]

        next_steps_np = np.array(next_steps[0])

        steps_to_add = min(forecast_horizon, n_steps - i)
        forecast[i:i + steps_to_add] = next_steps_np[:steps_to_add]

        # Update sequence for next step
        if i + steps_to_add < n_steps:
            # Shift sequence window and append new steps
            next_steps_jax = jnp.array(next_steps_np[:steps_to_add])[None,...]
            if devices:
                next_steps_jax = jax.device_put(next_steps_jax, device)

            new_sequence = jnp.concatenate([
                current_sequence[:, steps_to_add:, :],
                next_steps_jax
            ], axis=1)
            current_sequence = new_sequence

    return forecast








