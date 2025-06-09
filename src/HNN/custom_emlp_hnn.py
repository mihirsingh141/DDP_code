import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
from jax.experimental.ode import odeint
import jax
import random

SEED = 42
random.seed(SEED)
jax.random.PRNGKey(SEED)
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

    print(generator.mask)

    return np.stack([x.detach().cpu().numpy() for x in generator.getLi()],axis=1).squeeze()

# class HNN_EMLP(objax.Module):
#     def __init__(self, group, input_dim, hidden_dim=128, num_layers=3):
#         super().__init__()
#         self.G = group
        
#         # Define the EMLP architecture
#         rep_in = Vector(self.G)
#         rep_out = T(0)(self.G)  # Scalar output for Hamiltonian
        
#         self.hamiltonian = EMLP(
#             rep_in,
#             rep_out,
#             group=self.G,
#             num_layers=num_layers,
#             ch=hidden_dim,
#         )
    
#     def __call__(self, x):
#         """Compute the Hamiltonian value"""
#         return self.hamiltonian(x)
    
#     def time_derivative(self, x):
#         """Compute the Hamiltonian dynamics: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q"""
#         # Define single input gradient function
#         def H_single(x):
#             return self(x.reshape(1, -1)).sum()
        
#         # If batch, use vmap
#         if x.ndim > 1:
#             grad_H = vmap(grad(H_single))
#             dH = grad_H(x)
#         else:
#             dH = grad(H_single)(x)
        
#         # Split into q and p dimensions
#         q_dim = x.shape[-1] // 2
#         dH_dq = dH[..., :q_dim]
#         dH_dp = dH[..., q_dim:]
        
#         # Correct Hamiltonian equations
#         dq_dt = dH_dp   # dq/dt = ∂H/∂p
#         dp_dt = -dH_dq  # dp/dt = -∂H/∂q
        
#         return jnp.concatenate([dq_dt, dp_dt], axis=-1)

class HNN_EMLP(objax.Module):
    def __init__(self, group, input_dim, rep=1,hidden_dim=128, num_layers=3):
        super().__init__()
        self.G = group
        rep_in = Vector(self.G) * rep
        rep_out = T(0)(self.G)

        self.input_dim = input_dim
        
        if input_dim % 2 != 0:
            raise ValueError(f"Input dimension must be even for Hamiltonian system, got {input_dim}")
        
        # rep_in = Vector(SO(2)) * 4 
        # rep_out = T(0)(self.G)  # Scalar output for Hamiltonian
        
        self.hamiltonian = EMLP(
            rep_in,
            rep_out,
            group=self.G,
            num_layers=num_layers,
            ch=hidden_dim,
        )
    
    def __call__(self, x):
        """Compute the Hamiltonian value"""
        return self.hamiltonian(x)

    def time_derivative(self, x):
        """Compute the Hamiltonian dynamics: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q"""

        def H_scalar(xi):
            """Returns scalar Hamiltonian value for a single input xi"""
            return jnp.squeeze(self.hamiltonian(xi[None]))  # Ensures scalar output

        # Compute gradients of H wrt x
        if x.ndim == 1:
            # Single input
            dH = jax.grad(H_scalar)(x)
        else:
            # Batch input
            dH = jax.vmap(jax.grad(H_scalar))(x)

        # Split into q and p
        q_dim = x.shape[-1] // 2
        dH_dq = dH[..., :q_dim]
        dH_dp = dH[..., q_dim:]

        # Hamilton's equations
        dq_dt = dH_dp
        dp_dt = -dH_dq

        return jnp.concatenate([dq_dt, dp_dt], axis=-1)


def debug_model(model, test_input):
    """Debug function to check what's happening"""
    print("\n=== Debugging HNN Model ===")
    
    # Test single input
    x = jnp.array([1.0, 0.0], dtype=jnp.float32)  # [q, p]
    print(f"Test input: {x}")
    
    # Check Hamiltonian output
    H_value = model(x.reshape(1, -1))
    print(f"Hamiltonian value: {H_value}")
    
    # Check derivatives
    derivatives = model.time_derivative(x.reshape(1, -1))
    print(f"Derivatives: {derivatives}")
    
    # Check individual gradients
    def H_single(x):
        return model(x.reshape(1, -1)).sum()
    
    grad_H = grad(H_single)
    dH = grad_H(x)
    print(f"Gradient: {dH}")
    
    q_dim = x.shape[0] // 2
    dH_dq = dH[:q_dim]
    dH_dp = dH[q_dim:]
    print(f"dH/dq: {dH_dq}, dH/dp: {dH_dp}")
    print(f"dq/dt: {dH_dp}, dp/dt: {-dH_dq}")
    
    return derivatives     

def debug_simulation(model, initial_state, t_span, num_points=10):
    """Debug the simulation process"""
    print("\n=== Debugging Simulation ===")
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    # Convert initial state
    if hasattr(initial_state, 'device_buffer'):
        initial_state = np.array(initial_state)
    
    state = initial_state.copy()
    trajectory = [state]
    
    for i in range(num_points - 1):
        # Get derivatives at current state
        state_jax = jnp.array(state, dtype=jnp.float32).reshape(1, -1)
        derivatives = model.time_derivative(state_jax)
        derivatives = np.array(derivatives).flatten()
        
        print(f"Step {i}: state={state}, derivatives={derivatives}")
        
        # Simple Euler step
        dt = t_eval[i+1] - t_eval[i]
        state = state + derivatives * dt
        trajectory.append(state.copy())
    
    return np.array(trajectory)


    

def train_hnn_emlp(G, x, dxdt, test_x, test_dxdt, batch_size, epochs, lr, input_dim, rep=1, hidden_dim=128, num_layers=3, debug=False):
    import jax.random
    
    device = get_device()
    model = HNN_EMLP(G, input_dim, rep, hidden_dim, num_layers)

    # ---- Weight initialization tweak to prevent trivial constants ----
    rng = jax.random.PRNGKey(42)
    for k, v in model.vars().items():
        if 'W' in k or 'b' in k:
            init_noise = 0.01 * jax.random.normal(rng, v.value.shape)
            v.assign(v.value + init_noise)


    opt = objax.optimizer.Adam(model.vars())

    # --- Convert tensors ---
    to_numpy = lambda t: t.detach().cpu().numpy() if hasattr(t, 'detach') else t
    x, dxdt, test_x, test_dxdt = map(to_numpy, (x, dxdt, test_x, test_dxdt))
    x = np.array(x, dtype=np.float32)
    dxdt = np.array(dxdt, dtype=np.float32)
    test_x = np.array(test_x, dtype=np.float32)
    test_dxdt = np.array(test_dxdt, dtype=np.float32)

    # ---- Loss with regularization to discourage constant output ----
    @objax.Function.with_vars(model.vars())
    def loss_fn(states, true_derivatives):
        pred_derivatives = model.time_derivative(states)
        main_loss = jnp.mean((pred_derivatives - true_derivatives) ** 2)
        # H_vals = model(states)
        # reg_loss = -jnp.std(H_vals)
        # total = main_loss + 0.01 * reg_loss
        return main_loss

    grad_loss = objax.GradValues(loss_fn, model.vars())

    @objax.Function.with_vars(model.vars() + opt.vars())
    def train_op(states, true_derivatives):
        grads, values = grad_loss(states, true_derivatives)
        opt(lr, grads)
        g_norm = jnp.sqrt(sum([jnp.sum(g**2) for g in grads]))
        return values, g_norm

    loss_fn_jit = objax.Jit(loss_fn)
    train_op_jit = objax.Jit(train_op)

    def create_batches(X, y, batch_size, shuffle=True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        for i in range(0, n_samples, batch_size):
            yield X[indices[i:i+batch_size]], y[indices[i:i+batch_size]]

    def validate(data_x, data_dxdt, batch_size=batch_size):
        val_loss = 0.0
        num_batches = 0
        for batch_x, batch_dxdt in create_batches(data_x, data_dxdt, batch_size, shuffle=False):
            try:
                loss = loss_fn_jit(jnp.array(batch_x), jnp.array(batch_dxdt))
                val_loss += float(loss if not isinstance(loss, list) else loss[0])
                num_batches += 1
            except Exception as e:
                print(f"Validation error: {e}")
        return val_loss / num_batches if num_batches else float('inf')

    stats = {'train_loss': [], 'test_loss': [], 'grad_norm': []}
    best_test_loss = float('inf')
    best_vars = None

    print(f"Starting training: {epochs} epochs")

    for epoch in tqdm(range(epochs)):
        train_loss = 0.0
        train_grad_norm = 0.0
        num_batches = 0

        for i, (batch_x, batch_dxdt) in enumerate(create_batches(x, dxdt, batch_size)):
            try:
                batch_x_jax = jnp.array(batch_x)
                batch_dxdt_jax = jnp.array(batch_dxdt)
                loss, g_norm = train_op_jit(batch_x_jax, batch_dxdt_jax)

                if isinstance(loss, list): loss = loss[0]
                if isinstance(g_norm, list): g_norm = g_norm[0]

                train_loss += float(loss)
                train_grad_norm += float(g_norm)
                num_batches += 1

                
                if debug:
                    print(f"[Epoch {epoch} Batch {i}] Loss: {loss:.6f}, Grad Norm: {g_norm:.6f}")
            except Exception as e:
                print(f"Training error in batch {i}: {e}")

        if num_batches == 0:
            print("No valid batches this epoch")
            continue

        train_loss /= num_batches
        train_grad_norm /= num_batches
        test_loss = validate(test_x, test_dxdt)

        stats['train_loss'].append(train_loss)
        stats['test_loss'].append(test_loss)
        stats['grad_norm'].append(train_grad_norm)

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Grad Norm: {train_grad_norm:.6f}")

        # Early stopping if test loss very small
        if test_loss < 1e-6:
            print(f"Early stopping at epoch {epoch+1}, test loss {test_loss:.8f}")
            break

    return model, stats



def solve_ivp_with_nn_emlp(model,initial_state,t_span,t_eval=None,method='RK45',rtol=1e-8,atol=1e-8,debug=False):
    device = get_device()
    
    def nn_dynamics(t, state):
        """Function that returns the derivatives for the ODE solver"""
        try:
            # Convert to JAX array
            state_jax = jnp.array(state, dtype=jnp.float32).reshape(1, -1)
            
            # Compute derivatives using the HNN's time_derivative function
            derivatives = model.time_derivative(state_jax)
            
            # Convert back to numpy for scipy
            if hasattr(derivatives, 'device_buffer'):
                # For JAX arrays, use numpy() method
                return np.array(derivatives).flatten()
            else:
                # For regular numpy arrays
                return derivatives.flatten()
                
        except Exception as e:
            if debug:
                print(f"Error in dynamics function: {e}")
                print(f"State: {state}")
            # Return zeros as fallback
            return np.zeros_like(state)
    
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
    
    # Convert initial_state to numpy if it's a JAX array
    if hasattr(initial_state, 'device_buffer'):
        initial_state = np.array(initial_state)
    
    # Solve the ODE
    try:
        solution = solve_ivp(
            fun=nn_dynamics,
            t_span=t_span,
            y0=initial_state,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol
        )
        
        if not solution.success and debug:
            print(f"Warning: ODE solver did not converge. Message: {solution.message}")
            
        return solution
    except Exception as e:
        if debug:
            print(f"Error in ODE solver: {e}")
        # Create a dummy solution as fallback
        dummy_solution = type('obj', (object,), {
            't': t_eval,
            'y': np.tile(initial_state, (len(t_eval), 1)).T,
            'success': False,
            'message': f"Failed with error: {str(e)}"
        })
        return dummy_solution

def simulate_hnn_emlp(model, initial_state, t_span, num_points=1000, debug=False):
    device = get_device()
    # Create time points for evaluation
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    # Convert initial_state to numpy array if it's a JAX array
    if hasattr(initial_state, 'device_buffer'):
        initial_state = np.array(initial_state)
    initial_state = np.array(initial_state, dtype=np.float64)
    
    if debug:
        print(f"Starting simulation with initial state: {initial_state}")
    
    # Test the model on initial state to catch errors early
    # try:
    #     state_jax = jnp.array(initial_state, dtype=jnp.float32).reshape(1, -1)
    #     derivatives = model.time_derivative(state_jax)
    #     if debug:
    #         print(f"Initial derivatives: {np.array(derivatives)}")
    # except Exception as e:
    #     print(f"Error testing model on initial state: {e}")
    
    solution = solve_ivp_with_nn_emlp(
        model=model,
        initial_state=initial_state,
        t_span=t_span,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-8,
        debug=debug
    )
    
    t = solution.t
    y = solution.y  
    
    return t, y

def solve_ivp_with_nn_emlp_gpu(model, initial_state, t_span, t_eval=None, debug=False):
    """GPU-accelerated ODE solver using JAX"""
    
    if t_eval is None:
        t_eval = jnp.linspace(t_span[0], t_span[1], 1000)
    else:
        t_eval = jnp.array(t_eval)
    
    # Ensure initial_state is a JAX array on GPU
    initial_state = jnp.array(initial_state, dtype=jnp.float32)
    
    def nn_dynamics(state, t):
        """JAX-compatible dynamics function that stays on GPU"""
        try:
            # Reshape for model input (assuming model expects batch dimension)
            state_batch = state.reshape(1, -1)
            
            # Compute derivatives - this stays on GPU
            derivatives = model.time_derivative(state_batch)
            
            # Return flattened derivatives
            return derivatives.flatten()
            
        except Exception as e:
            if debug:
                print(f"Error in dynamics function: {e}")
            # Return zeros as fallback
            return jnp.zeros_like(state)
    
    try:
        # Use JAX's odeint - this runs entirely on GPU
        solution_y = odeint(nn_dynamics, initial_state, t_eval)
        
        # Create solution object similar to scipy's solve_ivp
        solution = type('Solution', (), {
            't': t_eval,
            'y': solution_y.T,  # Transpose to match scipy format
            'success': True,
            'message': 'Integration successful'
        })()
        
        if debug:
            print(f"GPU integration completed successfully")
            print(f"Solution shape: {solution.y.shape}")
            
        return solution
        
    except Exception as e:
        if debug:
            print(f"Error in GPU ODE solver: {e}")
        
        # Fallback solution
        dummy_solution = type('Solution', (), {
            't': t_eval,
            'y': jnp.tile(initial_state, (len(t_eval), 1)).T,
            'success': False,
            'message': f"Failed with error: {str(e)}"
        })()
        return dummy_solution

def simulate_hnn_emlp_gpu(model, initial_state, t_span, num_points=1000, debug=False):
    """GPU-accelerated HNN simulation"""
    
    # Create time points for evaluation on GPU
    t_eval = jnp.linspace(t_span[0], t_span[1], num_points)
    
    # Ensure initial_state is on GPU
    initial_state = jnp.array(initial_state, dtype=jnp.float32)
    
    if debug:
        print(f"Starting GPU simulation with initial state: {initial_state}")
        print(f"JAX device: {jax.devices()}")
    
    # Test the model on initial state
    if debug:
        try:
            state_batch = initial_state.reshape(1, -1)
            derivatives = model.time_derivative(state_batch)
            print(f"Initial derivatives: {derivatives}")
        except Exception as e:
            print(f"Error testing model on initial state: {e}")
    
    solution = solve_ivp_with_nn_emlp_gpu(
        model=model,
        initial_state=initial_state,
        t_span=t_span,
        t_eval=t_eval,
        debug=debug
    )
    
    return solution.t, solution.y

if __name__ == "__main__":
    device = get_device()
    n_dim = 2
    n_channel = 1
    G = get_generators(n_dim,n_channel,'saved_model/default/spring_mass_generator_99.pt')
    # G = CustomGroup(n_dim,lie_algebra) 

    input_dim=2
    hidden_dim=128
    num_layers=3

    model = HNN_EMLP(G,input_dim,hidden_dim,num_layers)

    # lie_algebra = lie_algebra.reshape(n_dim,n_dim)
    print(model)
    

