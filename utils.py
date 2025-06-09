import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy, scipy.misc, scipy.integrate
from numpy import *
solve_ivp = scipy.integrate.solve_ivp

# coscorr = lambda x,y: np.trace(x.T@y)/np.norm(x)/np.norm(y)

cos_corr = lambda x,y: torch.trace(x.T @ y) / torch.norm(x) / torch.norm(y)

# scale the tensor to have dummy position equal to 1
def affine_coord(tensor, dummy_pos=None):
    # tensor: B*T*K
    if dummy_pos is not None:
        return tensor / tensor[..., dummy_pos].unsqueeze(-1)
    else:
        return tensor

# Lorentz group Lie algebra
L_lorentz = np.zeros((6,4,4))
k = 3
for i in range(3):
    for j in range(i):
        L_lorentz[k,i+1,j+1] = 1
        L_lorentz[k,j+1,i+1] = -1
        k += 1
for i in range(3):
    L_lorentz[i,1+i,0] = 1
    L_lorentz[i,0,1+i] = 1
L_e = np.zeros((6, 4, 4))
k = 3
for i in range(3):
    for j in range(i):
        L_e[k,i,j] = 1
        L_e[k,j,i] = -1
        k += 1
for i in range(3):
    L_e[i,i,3] = 1
L_lorentz = torch.tensor(L_lorentz, dtype=torch.float32)
def getLorentzLieAlgebra():
    return L_lorentz
def getEuclideanLieALgebra():
    return L_e

def randomSO13pTransform(x, var=1):
    L = getLorentzLieAlgebra().to(x.device)
    z = var * torch.randn(x.shape[0], 6).to(x.device)
    g_z = torch.matrix_exp(torch.einsum('cjk,bc->bjk', L, z))
    return torch.einsum('bij,bkj->bki', g_z, x)

def randomSO3Transform(x):
    L = getLorentzLieAlgebra()[3:, :, :].to(x.device)
    z = torch.randn(x.shape[0], 3).to(x.device)
    g_z = torch.matrix_exp(torch.einsum('cjk,bc->bjk', L, z))
    return torch.einsum('bij,bkj->bki', g_z, x)

# def integrate_model(model,t_span,y0,fun=None,**kwargs):
#     def default_fun(t,np_x):
#         x = torch.tensor(np_x,requires_grad=True,dtype=torch.float32)
#         x = x.view(1,np.size(np_x)) 
#         dx = model.time_derivative(x).data.numpy().reshape(-1)
#         return dx 
#     fun = default_fun if fun is None else fun
#     return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

# def rk4(fun, y0, t, dt, *args, **kwargs):
#   dt2 = dt / 2.0
#   k1 = fun(y0, t, *args, **kwargs)
#   k2 = fun(y0 + dt2 * k1, t + dt2, *args, **kwargs)
#   k3 = fun(y0 + dt2 * k2, t + dt2, *args, **kwargs)
#   k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)
#   dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
#   return dy

# def choose_nonlinearity(name):
#   nl = None
#   if name == 'tanh':
#     nl = torch.tanh
#   elif name == 'relu':
#     nl = torch.relu
#   elif name == 'sigmoid':
#     nl = torch.sigmoid
#   elif name == 'softplus':
#     nl = torch.nn.functional.softplus
#   elif name == 'selu':
#     nl = torch.nn.functional.selu
#   elif name == 'elu':
#     nl = torch.nn.functional.elu
#   elif name == 'swish':
#     nl = lambda x: x * torch.sigmoid(x)
#   else:
#     raise ValueError("nonlinearity not recognized")
#   return nl

import torch
import torch.nn.functional as F

def choose_nonlinearity(name):
    """Choose a nonlinearity for neural networks
    
    Args:
        name: Name of the nonlinearity
        
    Returns:
        Nonlinearity function
    """
    if name == 'tanh':
        return torch.tanh
    elif name == 'relu':
        return F.relu
    elif name == 'sigmoid':
        return torch.sigmoid
    elif name == 'softplus':
        return F.softplus
    elif name == 'selu':
        return F.selu
    elif name == 'elu':
        return F.elu
    elif name == 'swish':
        return lambda x: x * torch.sigmoid(x)
    else:
        return F.relu

def rk4(fun, y0, t, dt):
    """Fourth-order Runge-Kutta integration step
    
    Args:
        fun: Function to integrate (returns derivative)
        y0: Initial state
        t: Current time (for time-dependent systems)
        dt: Time step
        
    Returns:
        Next state
    """
    k1 = fun(y0, t)
    k2 = fun(y0 + dt * k1 / 2, t + dt / 2)
    k3 = fun(y0 + dt * k2 / 2, t + dt / 2)
    k4 = fun(y0 + dt * k3, t + dt)
    return y0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def to_complex(x):
    """Convert tensor to complex representation
    
    Args:
        x: Input tensor
        
    Returns:
        Complex tensor
    """
    return torch.complex(x[..., 0], x[..., 1])

def from_complex(x):
    """Convert complex tensor to real representation
    
    Args:
        x: Complex tensor
        
    Returns:
        Real tensor with extra dimension
    """
    return torch.stack([x.real, x.imag], dim=-1)

def angular_frequency(k, m):
    """Calculate angular frequency for spring-mass system
    
    ω = sqrt(k/m)
    
    Args:
        k: Spring constant
        m: Mass
        
    Returns:
        Angular frequency
    """
    return (k / m) ** 0.5

def energy_conservation_error(hamiltonian_trajectory):
    """Calculate energy conservation error
    
    Args:
        hamiltonian_trajectory: Energy over time
        
    Returns:
        Relative energy drift
    """
    return abs(hamiltonian_trajectory.max() - hamiltonian_trajectory.min()) / hamiltonian_trajectory.mean()

def plot_phase_portrait(model, q_range, p_range, num_points=20, title="Phase Portrait"):
    """Plot vector field of the dynamics
    
    Args:
        model: HNN model
        q_range: Range for position [q_min, q_max]
        p_range: Range for momentum [p_min, p_max]
        num_points: Number of grid points per dimension
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    q = np.linspace(q_range[0], q_range[1], num_points)
    p = np.linspace(p_range[0], p_range[1], num_points)
    
    Q, P = np.meshgrid(q, p)
    
    dqdt = np.zeros_like(Q)
    dpdt = np.zeros_like(P)
    
    for i in range(num_points):
        for j in range(num_points):
            state = torch.tensor([[Q[i, j], P[i, j]]], dtype=torch.float32)
            state.requires_grad_(True)
            
            with torch.enable_grad():
                deriv = model.time_derivative(state)
                dqdt[i, j] = deriv[0, 0].item()
                dpdt[i, j] = deriv[0, 1].item()
    
    plt.figure(figsize=(10, 8))
    plt.streamplot(Q, P, dqdt, dpdt, density=1.5, color='black', linewidth=1)
    plt.xlabel('Position (q)')
    plt.ylabel('Momentum (p)')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    
    return plt