import matplotlib.pyplot as plt
import pickle
import torch
import os
from scipy.integrate import solve_ivp
from tqdm import tqdm
import numpy as np

class LorenzSimulator:
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        """
        Initialize a Lorenz system simulator
        
        Parameters:
        -----------
        sigma : float
            First parameter of the Lorenz system
        rho : float
            Second parameter of the Lorenz system
        beta : float
            Third parameter of the Lorenz system
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def derivatives(self, t, state):
        """
        Calculate derivatives for the Lorenz system
        
        Parameters:
        -----------
        t : float
            Time (not used in autonomous systems)
        state : ndarray
            System state [x, y, z]
            
        Returns:
        --------
        derivatives : ndarray
            Derivatives [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        
        return [dx_dt, dy_dt, dz_dt]
    
    def simulate(self, t_span, initial_state=None, t_eval=None):
        """
        Simulate the Lorenz system
        
        Parameters:
        -----------
        t_span : tuple
            (t_start, t_end)
        initial_state : ndarray or None
            Initial state [x, y, z]
            If None, random initial values are used
        t_eval : ndarray or None
            Times at which to evaluate the solution
            
        Returns:
        --------
        t : ndarray
            Time points
        y : ndarray
            System states at each time point [x, y, z]
        """
        if initial_state is None:
            # Random initial values
            initial_x = np.random.uniform(-15, 15)
            initial_y = np.random.uniform(-15, 15)
            initial_z = np.random.uniform(0, 40)
            initial_state = [initial_x, initial_y, initial_z]
        
        # Solve the ODE
        solution = solve_ivp(
            fun=self.derivatives,
            t_span=t_span,
            y0=initial_state,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-6
        )
        
        # Reshape to [timesteps, features]
        return solution.t, np.array([solution.y[0], solution.y[1], solution.y[2]]).T


def generate_lorenz_dataset(
    save_path='Data/lorenz_dataset.pkl',
    num_trajectories=100,
    t_span=(0, 5),
    num_steps=500,
    test_split=0.2,
    sigma=10.0,
    rho=28.0,
    beta=8.0/3.0,
    random_seed=42
):
    """
    Generate a dataset of Lorenz system trajectories
    
    Parameters:
    -----------
    save_path : str
        Path to save the pickle file
    num_trajectories : int
        Number of trajectories to generate
    t_span : tuple
        (t_start, t_end)
    num_steps : int
        Number of time steps per trajectory
    test_split : float
        Fraction of trajectories to use for testing
    sigma, rho, beta : float
        Lorenz system parameters
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    data_dict : dict
        Dictionary with training and test data
    """
    np.random.seed(random_seed)
    
    # Create time points for evaluation
    t_eval = np.linspace(t_span[0], t_span[1], num_steps)
    
    # Initialize arrays to store coordinates (x, y, z)
    all_coords = np.zeros((num_trajectories, num_steps, 3))
    
    # Create simulator once with fixed parameters
    simulator = LorenzSimulator(sigma=sigma, rho=rho, beta=beta)
    
    # Generate trajectories with different initial conditions
    for i in tqdm(range(num_trajectories), desc="Generating Lorenz Trajectories"):
        # Generate a random initial state
        initial_x = np.random.uniform(-15, 15)
        initial_y = np.random.uniform(-15, 15)
        initial_z = np.random.uniform(5, 35)
        initial_state = [initial_x, initial_y, initial_z]
        
        # Simulate
        _, states = simulator.simulate(t_span, initial_state, t_eval)
        
        # Store in our array
        all_coords[i] = states
    
    # Split into train and test
    num_test = int(num_trajectories * test_split)
    num_train = num_trajectories - num_test
    
    train_coords = all_coords[:num_train]
    test_coords = all_coords[num_train:]
    
    # Create data dictionary
    data_dict = {
        'coords': train_coords,
        'test_coords': test_coords,
        'metadata': {
            'sigma': sigma,
            'rho': rho,
            'beta': beta,
            't_span': t_span,
            'num_steps': num_steps
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as pickle
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"Dataset saved to {save_path}")
    return data_dict


class LorenzDataset(torch.utils.data.Dataset):
    def __init__(self, save_path='Data/lorenz_dataset.pkl', mode='train', 
                 input_timesteps=10, output_timesteps=1, flatten=False):
        """
        Dataset class for Lorenz system trajectories
        
        Parameters:
        -----------
        save_path : str
            Path to the pickle file with data
        mode : str
            'train' or 'test'
        input_timesteps : int
            Number of input timesteps for prediction
        output_timesteps : int
            Number of output timesteps to predict
        flatten : bool
            Whether to flatten the input and output sequences
        """
        # Load data
        with open(save_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Select train or test data
        if mode == 'train':
            self.data = self.data['coords']
        else:
            self.data = self.data['test_coords']
        
        # Get data dimensions
        self.feat_dim = self.data.shape[2]  # Should be 3 (x, y, z)
        
        # Prepare input-output sequences
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        
        self.X, self.y = [], []
        self.N = self.data.shape[0]  # Number of trajectories
        trj_timesteps = self.data.shape[1]  # Timesteps per trajectory
        
        # For each trajectory, create sliding windows
        for i in range(self.N):
            for t in range(trj_timesteps - input_timesteps - output_timesteps + 1):
                self.X.append(self.data[i, t:t+input_timesteps, :])
                self.y.append(self.data[i, t+input_timesteps:t+input_timesteps+output_timesteps, :])
        
        # Convert to PyTorch tensors
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
        
        # Get dataset length
        self.len = self.X.shape[0]
        
        # Flatten if requested
        if flatten:
            self.X = self.X.reshape(self.len, -1)
            self.y = self.y.reshape(self.len, -1)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

