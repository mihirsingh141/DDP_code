import matplotlib.pyplot as plt
import pickle
import torch
import os
from scipy.integrate import solve_ivp
from tqdm import tqdm
import numpy as np

class SpringMassSimulator:
    def __init__(self, mass=1.0, k=1.0, damping=0.0, external_force=0.0):
        """
        Initialize a simple spring-mass system simulator
        
        Parameters:
        -----------
        mass : float
            Mass of the object
        k : float
            Spring constant
        damping : float
            Damping coefficient
        external_force : float
            External force applied to the mass
        """
        self.mass = mass
        self.k = k
        self.damping = damping
        self.external_force = external_force
    
    def derivatives(self, t, state):
        """
        Calculate derivatives for the system (dx/dt = v, dv/dt = a)
        
        Parameters:
        -----------
        t : float
            Time (not used in time-invariant systems)
        state : ndarray
            System state [position, velocity]
            
        Returns:
        --------
        derivatives : ndarray
            Derivatives [velocity, acceleration]
        """
        x, v = state
        
        # Spring force: F = -kx
        spring_force = -self.k * x
        
        # Damping force: F = -cv
        damping_force = -self.damping * v
        
        # Total force: F = -kx - cv + F_ext
        total_force = spring_force + damping_force + self.external_force
        
        # Acceleration: a = F/m
        acceleration = total_force / self.mass
        
        return [v, acceleration]
    
    def simulate(self, t_span, initial_state=None, num_points=1000, t_eval=None):
        """
        Simulate the system
        
        Parameters:
        -----------
        t_span : tuple
            (t_start, t_end)
        initial_state : ndarray or None
            Initial state [position, velocity]
            If None, random initial position between 0.2-0.8 and zero velocity
        t_eval : ndarray or None
            Times at which to evaluate the solution
            
        Returns:
        --------
        t : ndarray
            Time points
        y : ndarray
            System states at each time point [positions, velocities]
        """
        if initial_state is None:
            # Random initial position between 0.2 and 0.8
            initial_position = np.random.uniform(0.2, 0.8)
            # Zero initial velocity
            initial_velocity = 0.0
            initial_state = [initial_position, initial_velocity]
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0],t_span[1],num_points)

        # Solve the ODE
        solution = solve_ivp(
            fun=self.derivatives,
            t_span=t_span,
            y0=initial_state,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-8
        )
        
        # Reshape to [timesteps, features]
        return solution.t, np.array([solution.y[0], solution.y[1]]).T


def generate_spring_mass_dataset(
    save_path='Data/spring_mass_dataset.pkl',
    num_trajectories=1,
    t_span=(0, 10),
    num_steps=100,
    test_split=0.2,
    mass=1.0,
    k=1.0,
    damping=0.0,
    random_seed=42,
    num_points=1000
):
    """
    Generate a dataset of simple spring-mass system trajectories
    
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
    mass : float
        Mass of the object
    k : float
        Spring constant
    damping : float
        Damping coefficient
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
    
    # Initialize arrays to store coordinates (position and velocity)
    all_coords = np.zeros((num_trajectories, num_steps, 2))
    
    # Create simulator once with fixed parameters
    simulator = SpringMassSimulator(mass=mass, k=k, damping=damping)
    
    # Generate trajectories with different initial conditions
    for i in tqdm(range(num_trajectories), desc="Generating Trajectories"):
        # Generate a random initial state
        initial_position = np.random.uniform(0.2, 0.8)
        initial_velocity = np.random.uniform(-0.1, 0.1)
        initial_state = [initial_position, initial_velocity]
        
        # Simulate
        _, states = simulator.simulate(t_span, initial_state, num_points, t_eval)
        
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
            'mass': mass,
            'k': k,
            'damping': damping,
            't_span': t_span,
            'num_steps': num_steps
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as pickle
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    return data_dict


class SpringMassDataset(torch.utils.data.Dataset):
    def __init__(self, save_path='Data/spring_mass_dataset.pkl', mode='train', 
                 trj_timesteps=100, input_timesteps=4, output_timesteps=1, 
                 flatten=False):
        """
        Dataset class for spring-mass system trajectories
        
        Parameters:
        -----------
        save_path : str
            Path to the pickle file with data
        mode : str
            'train' or 'test'
        trj_timesteps : int
            Number of timesteps in each trajectory
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
        self.feat_dim = self.data.shape[2]  # Should be 2 (position, velocity)
        
        # Prepare input-output sequences
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        
        self.X, self.y = [], []
        self.N = self.data.shape[0]  # Number of trajectories
        trj_timesteps = self.data.shape[1]  # Timesteps per trajectory
        
        # For each trajectory, create sliding windows
        for i in range(self.N):
            for t in range(trj_timesteps - input_timesteps - output_timesteps):
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

