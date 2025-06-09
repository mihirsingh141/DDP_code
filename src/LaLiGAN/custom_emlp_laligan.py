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
from src.NN.custom_emlp_multistep import PredModel_EMLP
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


def train_predModel_laligan(G, jax_autoencoder, train_dataset, test_dataset, batch_size, epochs, lr, n_dim, input_dim, output_dim, hidden_dim=128, num_layers=3, method=1):
    model = PredModel_EMLP(G, n_dim, input_dim, hidden_dim, output_dim, num_layers)
    opt = objax.optimizer.Adam(model.vars())

    train_loader = DataLoader(train_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)
    

    @objax.Jit
    @objax.Function.with_vars(model.vars())
    def loss(zx,y):
        bs = zx.shape[0]
        zy_hat = model(zx)
        zy_hat = zy_hat.reshape(bs,-1,n_dim)
        y_hat = jax_autoencoder.decode(zy_hat,training=False)
        y_hat = y_hat.reshape(bs,-1) 

        return jnp.mean((y_hat - y) ** 2)
    

    grad_value_func= objax.GradValues(loss, model.vars())

    @objax.Jit 
    @objax.Function.with_vars(model.vars()+opt.vars())
    def train_op(zx, zy, lr):    
        g, v = grad_value_func(zx, zy)
        opt(lr=lr, grads=g)
        return g, v
    
    

    def evaluate_model(loader):
        total_loss = 0.0
        num_batches = 0
        for x, y in loader:
            # Convert PyTorch tensors to NumPy arrays
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            
            # Encode with JAX autoencoder
            zx = jax_autoencoder.encode(jnp.array(x_np),training=False)
            zx = zx.reshape(x.shape[0], -1)

            y_jax = jnp.array(y_np.reshape(x.shape[0], -1))
            
            # Compute loss
            batch_loss = loss(zx, y_jax)
            total_loss += float(batch_loss)
            num_batches += 1

        return total_loss / num_batches

    print(f"Starting training: {epochs} epochs")

    stats = {'train_loss': [], 'test_loss': [], 'grad_norm': []}

    for epoch in tqdm(range(epochs)):
        train_loss = 0.0
        train_grad_norm = 0.0
        num_batches = 0

        for i, (x, y) in enumerate(train_loader):
            # Convert to NumPy arrays
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            
            # Encode with JAX autoencoder
            zx = jax_autoencoder.encode(jnp.array(x_np),training=False)
            zx = zx.reshape(x.shape[0], -1)

            y_jax = jnp.array(y_np.reshape(x.shape[0], -1))

            try:
                # Train with JAX arrays
                g, batch_loss_val = train_op(zx, y_jax, lr)

                g_norm = sum(jnp.sum(jnp.square(gi)) for gi in g) ** 0.5
                train_grad_norm += float(g_norm)

                if isinstance(batch_loss_val, list):
                    batch_loss_val = batch_loss_val[0].mean()
                train_loss += float(batch_loss_val)

                num_batches += 1

                
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue

        if num_batches == 0:
            print("No valid batches in epoch.")
            continue

        train_loss /= num_batches
        train_grad_norm /= num_batches
        stats['train_loss'].append(train_loss)
        stats['grad_norm'].append(train_grad_norm)

        test_loss = evaluate_model(test_loader)
        stats['test_loss'].append(test_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

        if test_loss < 1e-6:
            print(f"Early stopping at epoch {epoch+1} - test loss: {test_loss:.8f}")
            break

    return model, stats


def forecast_emlp_laligan(model, autoencoder, last_sequence, n_steps, feature_dim, latent_dim):
    devices = jax.devices("gpu")
    if devices:
        device = devices[0]
        sequence = jax.device_put(jnp.array(last_sequence)[None, ...], device)
    else:
        sequence = jnp.array(last_sequence)[None, ...]

    # Encode the input sequence to latent space
    current_sequence_enc = autoencoder.encode(sequence, training=False)

    # JIT compile the model
    model_jit = jax.jit(model)

    # Run a forward pass to get forecast horizon
    sample_pred = model_jit(current_sequence_enc).reshape(1,-1,latent_dim)
    forecast_horizon = sample_pred.shape[1]
    # Prepare forecast array in observation space

    forecast = np.zeros((n_steps, feature_dim))



    for i in range(0, n_steps, forecast_horizon):
        next_steps_enc = model_jit(current_sequence_enc).reshape(-1,forecast_horizon,latent_dim)
        if next_steps_enc.ndim == 2:
            next_steps_enc = next_steps_enc[:, None, :]

        # Decode to observation space
        next_steps = autoencoder.decode(next_steps_enc, training=False)
        next_steps_np = np.array(next_steps[0])  # Remove batch dimension

        # Store into forecast
        steps_to_add = min(forecast_horizon, n_steps - i)
        forecast[i:i + steps_to_add] = next_steps_np[:steps_to_add]

        if i + forecast_horizon < n_steps:
            # Re-encode next steps for next iteration
            next_steps_jax = jnp.array(next_steps_np[:steps_to_add])
            if devices:
                next_steps_jax = jax.device_put(next_steps_jax, device)

            next_steps_enc = autoencoder.encode(next_steps_jax[None, ...], training=False)

            # Shift and append to form new input sequence
            current_sequence_enc = jnp.concatenate([
                current_sequence_enc[:, forecast_horizon:, :],
                next_steps_enc
            ], axis=1)

    return forecast