import torch.nn as nn 
from torch.autograd.functional import jvp 
from torch.nn.utils.parametrizations import orthogonal

import objax
import objax.nn as obnn
import jax.numpy as jnp
import objax.functional as F


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape(self.shape)


class AutoEncoder(nn.Module):
    '''
    Arguments:
        input_dim: dimension of input
        hidden_dim: dimension of hidden layer
        latent_dim: dimension of latent layer
        n_layers: number of hidden layers
        n_comps: number of components
        activation: activation function
        flatten: whether to flatten input
    Input:
        x: (batch_size, n_comps, input_dim)
    Output:
        z: (batch_size, n_comps, latent_dim)
        xhat: (batch_size, n_comps, input_dim)
    '''

    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, activation='ReLU', batch_norm=True, flatten=False, **kwargs):
        super().__init__() 
        self.flatten = nn.Flatten() if flatten else nn.Identity() 

        # Define activation functions
        activations = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid()
        }
        
        # Get the activation function, default to ReLU if not found
        self.activation = activations.get(activation, nn.ReLU())
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            self.activation,
            *[nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim), 
                nn.BatchNorm1d(hidden_dim), 
                self.activation
            ) for _ in range(n_layers-1)],

            nn.Linear(hidden_dim,latent_dim),  
            nn.BatchNorm1d(latent_dim)
        ) 

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,hidden_dim), 
            self.activation, 
            *[nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim), 
                self.activation
            ) for _ in range(n_layers)], 
            nn.Linear(hidden_dim,input_dim)
        )

    # def forward(self,x): 
    #     x = self.flatten(x) 
    #     z = self.encoder(x) 
    #     xhat = self.decoder(z) 

    #     return z, xhat 
    
    # def decode(self,z):
    #     return self.decoder(z) 
    
    def encode(self, x):
        batch_size, timesteps, features = x.shape
        
        # Reshape for processing each timestep
        x_reshaped = x.reshape(-1, features)
        
        # Apply encoder layers
        z = x_reshaped
        for layer in self.encoder:
            if isinstance(layer, nn.BatchNorm1d):
                z = layer(z)
            else:
                z = layer(z)
                
        # Reshape back to (batch, timesteps, latent_dim)
        z = z.reshape(batch_size, timesteps, self.latent_dim)
        return z

    def forward(self, x):
        batch_size, timesteps, features = x.shape
        
        # Encode
        z = self.encode(x)
        
        # Decode (apply decoder to each timestep)
        z_reshaped = z.reshape(-1, self.latent_dim)
        xhat_reshaped = self.decoder(z_reshaped)
        xhat = xhat_reshaped.reshape(batch_size, timesteps, self.input_dim)
        
        return z, xhat
    
    def decode(self, z):
        batch_size, timesteps, latent_dim = z.shape
        z_reshaped = z.reshape(-1, latent_dim)
        xhat_reshaped = self.decoder(z_reshaped)
        xhat = xhat_reshaped.reshape(batch_size, timesteps, self.input_dim)
        return xhat

    def compute_dz(self,x,dx):
        dz = jvp(self.encoder, x, v=dx) 
        return dz

    def compute_dx(self,z,dz):
        dx = jvp(self.decoder,z,v=dz) 
        return dx 

        

class JAXAutoEncoder(objax.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, activation='ReLU', batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.latent_dim = latent_dim

        activations = {
            'ReLU': F.relu,
            'LeakyReLU': F.leaky_relu,
            'Tanh': F.tanh,
            'Sigmoid': F.sigmoid
        }
        act_fn = activations.get(activation, F.relu)

        # Encoder
        encoder_layers = [obnn.Linear(input_dim, hidden_dim)]
        if batch_norm:
            encoder_layers.append(obnn.BatchNorm((hidden_dim,), redux=(0,), momentum=0.0, eps=1e-5))
        encoder_layers.append(act_fn)

        for _ in range(n_layers - 1):
            encoder_layers.append(obnn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                encoder_layers.append(obnn.BatchNorm((hidden_dim,), redux=(0,), momentum=0.0, eps=1e-5))
            encoder_layers.append(act_fn)

        encoder_layers.append(obnn.Linear(hidden_dim, latent_dim))
        if batch_norm:
            encoder_layers.append(obnn.BatchNorm((latent_dim,), redux=(0,), momentum=0.0, eps=1e-5))

        self.encoder = obnn.Sequential(encoder_layers)

        # Decoder
        decoder_layers = [obnn.Linear(latent_dim, hidden_dim), act_fn]
        for _ in range(n_layers):
            decoder_layers.append(obnn.Linear(hidden_dim, hidden_dim))
            decoder_layers.append(act_fn)

        decoder_layers.append(obnn.Linear(hidden_dim, input_dim))
        self.decoder = obnn.Sequential(decoder_layers)

    def encode(self, x, training=True):
        b, t, f = x.shape
        x_flat = x.reshape(-1, f)

        z = x_flat
        for layer in self.encoder:
            if isinstance(layer, obnn.BatchNorm):
                z = layer(z, training=training)
            else:
                z = layer(z)
        return z.reshape(b, t, self.latent_dim)

    def decode(self, z, training=True):
        b, t, f = z.shape
        z_flat = z.reshape(-1, f)

        x_hat = z_flat
        for layer in self.decoder:
            if isinstance(layer, obnn.BatchNorm):
                x_hat = layer(x_hat, training=training)
            else:
                x_hat = layer(x_hat)
        return x_hat.reshape(b, t, -1)

    def __call__(self, x, training=True):
        z = self.encode(x, training=training)
        x_hat = self.decode(z, training=training)
        return z, x_hat



def extract_weights_from_torch(autoencoder):
    weights = []

    def extract(seq):
        for layer in seq:
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.Linear):
                        w = sublayer.weight.detach().cpu().numpy().T
                        b = sublayer.bias.detach().cpu().numpy()
                        weights.append(('linear', w, b))
                    elif isinstance(sublayer, nn.BatchNorm1d):
                        gamma = sublayer.weight.detach().cpu().numpy()
                        beta = sublayer.bias.detach().cpu().numpy()
                        mean = sublayer.running_mean.detach().cpu().numpy()
                        var = sublayer.running_var.detach().cpu().numpy()
                        weights.append(('batchnorm', gamma, beta, mean, var))
            elif isinstance(layer, nn.Linear):
                w = layer.weight.detach().cpu().numpy().T
                b = layer.bias.detach().cpu().numpy()
                weights.append(('linear', w, b))
            elif isinstance(layer, nn.BatchNorm1d):
                gamma = layer.weight.detach().cpu().numpy()
                beta = layer.bias.detach().cpu().numpy()
                mean = layer.running_mean.detach().cpu().numpy()
                var = layer.running_var.detach().cpu().numpy()
                weights.append(('batchnorm', gamma, beta, mean, var))

    extract(autoencoder.encoder)
    extract(autoencoder.decoder)

    return weights

def load_weights_into_jax(jax_model, torch_weights):
    import jax.numpy as jnp  # Make sure to import this at the top
    
    idx = 0
    for layer in jax_model.encoder:
        if isinstance(layer, objax.nn.Linear):
            _, w, b = torch_weights[idx]
            # Convert NumPy arrays to JAX arrays
            layer.w.assign(jnp.array(w))
            layer.b.assign(jnp.array(b))
            idx += 1
        elif isinstance(layer, objax.nn.BatchNorm):
            _, gamma, beta, mean, var = torch_weights[idx]
            # Convert NumPy arrays to JAX arrays
            layer.gamma.assign(jnp.array(gamma))
            layer.beta.assign(jnp.array(beta))
            layer.running_mean.assign(jnp.array(mean))
            layer.running_var.assign(jnp.array(var))
            idx += 1

    for layer in jax_model.decoder:
        if isinstance(layer, objax.nn.Linear):
            _, w, b = torch_weights[idx]
            # Convert NumPy arrays to JAX arrays
            layer.w.assign(jnp.array(w))
            layer.b.assign(jnp.array(b))
            idx += 1
        elif isinstance(layer, objax.nn.BatchNorm):
            _, gamma, beta, mean, var = torch_weights[idx]
            # Convert NumPy arrays to JAX arrays
            layer.gamma.assign(jnp.array(gamma))
            layer.beta.assign(jnp.array(beta))
            layer.running_mean.assign(jnp.array(mean))
            layer.running_var.assign(jnp.array(var))
            idx += 1

    print("✅ All weights transferred successfully.")