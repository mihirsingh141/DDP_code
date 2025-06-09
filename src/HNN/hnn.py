import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp
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

class HNN(nn.Module):
    """Hamiltonian Neural Network
    
    Learns the Hamiltonian H(q, p) of a system from observed trajectories,
    and models dynamics that respect Hamilton's equations:
    dq/dt = ∂H/∂p
    dp/dt = -∂H/∂q
    """
    
    def __init__(self, input_dim, differentiable_model, field_type='conservative', baseline=False, assume_canonical_coords=True, device=None):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.field_type = field_type
        
        
        if assume_canonical_coords:
            if input_dim == 2:
                self.M = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32, device=device)
            else:
                self.M = torch.zeros(input_dim, input_dim, device=device)
                for i in range(input_dim // 2):
                    self.M[2*i, 2*i+1] = 1
                    self.M[2*i+1, 2*i] = -1
                self.M = self.M.float()
        else:
            self.M = self.permutation_tensor(input_dim)
    
    def forward(self, x):
        """Forward pass to compute the Hamiltonian"""
        if self.baseline:
            return self.differentiable_model(x)
        
        H = self.differentiable_model(x)
        
        if torch.isnan(H).any() or torch.isinf(H).any():
            print(f"Warning: NaN or Inf detected in Hamiltonian: {H}")
        
        return H
    
    
    def time_derivative(self, x, t=None, separate_fields=False, debug=False):
        """Compute the time derivatives dq/dt and dp/dt from the Hamiltonian
        
        For a Hamiltonian system:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q
        """
        device = get_device()
        
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=device)
            
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        
        x_tensor = x.detach().clone().to(device)
        x_tensor.requires_grad_(True)
        
        if debug:
            print(f"Input shape: {x_tensor.shape}")
            print(f"Requires grad: {x_tensor.requires_grad}")
        
        # Compute the Hamiltonian
        H = self.forward(x_tensor)
        
        if debug:
            print(f"Hamiltonian shape: {H.shape}")
        
        # Calculate gradients 
        try:
            dH = torch.autograd.grad(H.sum(), x_tensor, create_graph=True)[0]
        except Exception as e:
            print(f"Error computing gradient: {e}")
            print(f"Input tensor requires_grad: {x_tensor.requires_grad}")
            print(f"Hamiltonian: {H}")
            # Return zeros as a fallback
            return torch.zeros_like(x_tensor)
        
        if debug:
            print(f"Gradient ∂H/∂x shape: {dH.shape}")
            print(f"Gradient values: {dH}")
        
        # For a canonical system:
        # [dq/dt, dp/dt] = [∂H/∂p, -∂H/∂q]
        if self.assume_canonical_coords and x_tensor.shape[1] == 2:
            dq_dt = dH[:, 1]  # ∂H/∂p
            dp_dt = -dH[:, 0]  # -∂H/∂q
            derivatives = torch.stack([dq_dt, dp_dt], dim=1)
        else:
            # Use the symplectic matrix for general cases
            derivatives = dH @ self.M.to(dH.device)
        
        if debug:
            print(f"Time derivatives shape: {derivatives.shape}")
            print(f"Time derivatives: {derivatives}")
                
        return derivatives
        
    def permutation_tensor(self, n):
        """Constructs the Levi-Civita permutation tensor"""
        device = get_device()
        M = torch.ones(n, n, device=device)  # matrix of ones
        M *= 1 - torch.eye(n, device=device)  # clear diagonals
        M[::2] *= -1  # pattern of signs
        M[:, ::2] *= -1
    
        for i in range(n):  # make asymmetric
            for j in range(i+1, n):
                M[i, j] *= -1
        return M

class HNNDataset(torch.utils.data.Dataset):
    def __init__(self, states, derivatives):
        device = get_device()
        # Convert to tensors if not already
        if not isinstance(states, torch.Tensor):
            self.states = torch.tensor(states, dtype=torch.float32, device=device)
        else:
            self.states = states.to(device)
            
        if not isinstance(derivatives, torch.Tensor):
            self.derivatives = torch.tensor(derivatives, dtype=torch.float32, device=device)
        else:
            self.derivatives = derivatives.to(device)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.derivatives[idx]



def train_hnn(x, dxdt, test_x, test_dxdt, batch_size, epochs, lr, input_dim, hidden_dim, output_dim=1, debug=False):
    """Train the Hamiltonian Neural Network
    
    Args:
        x: Training states [batch, state_dim]
        dxdt: Training derivatives [batch, state_dim]
        test_x: Test states [batch, state_dim]
        test_dxdt: Test derivatives [batch, state_dim]
        batch_size: Mini-batch size
        epochs: Number of training epochs
        lr: Learning rate
        input_dim: Dimension of the state (e.g., 2 for a spring-mass system)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (should be 1 for Hamiltonian)
    
    Returns:
        model: Trained HNN model
        stats: Training statistics
    """
    device = get_device()
    print(f'Using Device : {device}')
    
    nn_model = MLP(input_dim, hidden_dim, output_dim, nonlinearity='tanh')
    model = HNN(input_dim, differentiable_model=nn_model, assume_canonical_coords=True,device=device)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Use a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    
    # Make sure input tensors are float32 and on the correct device
    x = x.to(device)
    dxdt = dxdt.to(device)
    test_x = test_x.to(device)
    test_dxdt = test_dxdt.to(device)
    
    # Create datasets and data loaders
    train_dataset = torch.utils.data.TensorDataset(x, dxdt)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_dxdt)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Training statistics
    stats = {'train_loss': [], 'test_loss': [], 'grad_norm': []}
    
    best_test_loss = float('inf')
    best_model = None
    
    print(f"Starting training: {epochs} epochs")
    
    for epoch in range(epochs):
        # Training
        model.train()

        epoch_grad_norm = 0
        train_loss = 0.0
        num_batches = 0
        
        for states, true_derivatives in train_loader:
            # Ensure states requires grad
            states = states.to(device).requires_grad_(True)
            true_derivatives = true_derivatives.to(device)

            
            # Forward pass to get predicted derivatives
            try:
                pred_derivatives = model.time_derivative(states)
            
                # Compute MSE loss
                loss = F.mse_loss(pred_derivatives, true_derivatives)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                
                # Compute gradient norm for monitoring
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                epoch_grad_norm += grad_norm
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
            
            if debug and num_batches % 10 == 0:
                print(f"Batch {num_batches}, Loss: {loss.item():.6f}, Grad norm: {grad_norm:.6f}")
        
        if num_batches == 0:
            print("No valid batches in training epoch")
            continue
            
        train_loss /= num_batches
        epoch_grad_norm /= num_batches
        stats['train_loss'].append(train_loss)
        stats['grad_norm'].append(grad_norm if 'grad_norm' in locals() else 0.0)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        num_test_batches = 0
        
        for states, true_derivatives in test_loader:
            # Ensure states requires grad for evaluation
            states = states.to(device).requires_grad_(True)
            true_derivatives = true_derivatives.to(device)
            
            # Forward pass
            try:
                with torch.enable_grad():  # Need this for computing derivatives
                    pred_derivatives = model.time_derivative(states)
                
                # Compute loss
                loss = F.mse_loss(pred_derivatives, true_derivatives)
                test_loss += loss.item()
                num_test_batches += 1
            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue
        
        if num_test_batches == 0:
            print("No valid batches in evaluation epoch")
            test_loss = float('inf')
        else:
            test_loss /= num_test_batches
            
        stats['test_loss'].append(test_loss)
        
        # Learning rate scheduler step
        scheduler.step(test_loss)
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        # Save best model
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

def solve_ivp_with_nn(model, initial_state, t_span, t_eval=None, method='RK45', rtol=1e-5, atol=1e-5, debug=False):
    """Use scipy's ODE solver with the HNN for simulation
    
    Args:
        model: Trained HNN model
        initial_state: Initial state of the system [q0, p0]
        t_span: (t_start, t_end) for simulation
        method: ODE solver method
        
    Returns:
        solution: ODE solution object
    """
    device = get_device()
    model.eval()  # Set model to evaluation mode
    
    def nn_dynamics(t, state):
        """Function that returns the derivatives for the ODE solver"""
        # Convert to tensor
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
            state_tensor.requires_grad_(True)  
            
            # Compute derivatives using the HNN
            with torch.enable_grad(): 
                try:
                    derivatives = model.time_derivative(state_tensor, debug=debug)
                    return derivatives.detach().cpu().numpy().flatten()
                except Exception as e:
                    print(f"Error in dynamics function: {e}")
                    print(f"State: {state}")
                    # Return zeros as fallback
                    return np.zeros_like(state)
    
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
    
    if torch.is_tensor(initial_state):
        initial_state = initial_state.detach().cpu().numpy()
    
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
        
        if not solution.success:
            print(f"Warning: ODE solver did not converge. Message: {solution.message}")
            
        return solution
    except Exception as e:
        print(f"Error in ODE solver: {e}")
        dummy_solution = type('obj', (object,), {
            't': t_eval,
            'y': np.tile(initial_state, (1, len(t_eval))),
            'success': False,
            'message': f"Failed with error: {str(e)}"
        })
        return dummy_solution

def simulate_hnn(model, initial_state, t_span, num_points=1000, debug=False):
    """Simulate system dynamics using trained HNN
    
    Args:
        model: Trained HNN model
        initial_state: Initial state [q0, p0]
        t_span: (t_start, t_end) for simulation
        num_points: Number of time points
        
    Returns:
        t: Time points
        y: System states at each time point [q_trajectory, p_trajectory]
    """
    device = get_device()
    # Create time points for evaluation
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    # Check if initial_state is tensor and convert if needed
    if torch.is_tensor(initial_state):
        initial_state = initial_state.detach().cpu().numpy()
    initial_state = np.array(initial_state, dtype=np.float64)
    
    
    # Test the model on initial state to catch errors early
    try:
        state_tensor = torch.tensor(initial_state, dtype=torch.float32, device=device).reshape(1, -1)
        state_tensor.requires_grad_(True)
        with torch.enable_grad():
            derivatives = model.time_derivative(state_tensor, debug=False)
    except Exception as e:
        print(f"Error testing model on initial state: {e}")
    
    solution = solve_ivp_with_nn(
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

def compute_derivatives_torch(q_trajectory, p_trajectory, dt):
    """Compute time derivatives from position and momentum trajectories
    
    Args:
        q_trajectory: Position trajectory [batch, time_steps]
        p_trajectory: Momentum trajectory [batch, time_steps]
        dt: Time step
        
    Returns:
        dq_dt: Position derivatives [batch, time_steps]
        dp_dt: Momentum derivatives [batch, time_steps]
    """
    device = get_device()
    # Convert to tensors if not already
    if not isinstance(q_trajectory, torch.Tensor):
        q_trajectory = torch.tensor(q_trajectory, dtype=torch.float32, device=device)
        p_trajectory = torch.tensor(p_trajectory, dtype=torch.float32, device=device)
    
    dq_dt = []
    dp_dt = []
    
    for i in range(len(q_trajectory)):
        dq_dt.append(torch.gradient(q_trajectory[i], spacing=dt)[0])
        dp_dt.append(torch.gradient(p_trajectory[i], spacing=dt)[0])
    
    return torch.stack(dq_dt), torch.stack(dp_dt)
