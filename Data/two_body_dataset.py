import matplotlib.pyplot as plt
import pickle
import torch
import os
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


class TwoBodySimulator:
    def __init__(self, G=1.0, m1=1.0, m2=1.0):
        """
        Initialize a two-body gravitational system simulator
        
        Parameters:
        -----------
        G : float
            Gravitational constant
        m1 : float
            Mass of first body
        m2 : float
            Mass of second body
        """
        self.G = G
        self.m1 = m1
        self.m2 = m2
        self.total_mass = m1 + m2
        self.reduced_mass = (m1 * m2) / self.total_mass
    
    def derivatives(self, t, state):
        """
        Calculate derivatives for the system
        
        Parameters:
        -----------
        t : float
            Time (not used in time-invariant systems)
        state : ndarray
            System state [x1, y1, x2, y2, px1, py1, px2, py2]
            where (x1, y1) and (x2, y2) are positions
            and (px1, py1) and (px2, py2) are momenta
            
        Returns:
        --------
        derivatives : ndarray
            Derivatives [dx1/dt, dy1/dt, dx2/dt, dy2/dt, dpx1/dt, dpy1/dt, dpx2/dt, dpy2/dt]
        """
        # Unpack the state vector
        x1, y1, x2, y2, px1, py1, px2, py2 = state
        
        # Calculate velocities from momenta
        vx1 = px1 / self.m1
        vy1 = py1 / self.m1
        vx2 = px2 / self.m2
        vy2 = py2 / self.m2
        
        # Calculate the distance between the bodies
        r_vec = np.array([x2 - x1, y2 - y1])
        r = np.sqrt(np.sum(r_vec**2))
        
        # Prevent division by zero
        if r < 1e-10:
            r = 1e-10
            
        # Calculate the gravitational force
        force_magnitude = self.G * self.m1 * self.m2 / r**3
        force_x = force_magnitude * r_vec[0]
        force_y = force_magnitude * r_vec[1]
        
        # Calculate acceleration using F = ma
        ax1 = force_x / self.m1
        ay1 = force_y / self.m1
        ax2 = -force_x / self.m2  # Opposite force on the second body
        ay2 = -force_y / self.m2
        
        # Return the derivatives
        return [vx1, vy1, vx2, vy2, force_x, force_y, -force_x, -force_y]
    
    def generate_initial_state(self, min_radius=1.0, max_radius=5.0, eccentricity_range=(0.0, 0.7), orbit_noise=0.0):
        """
        Generate a random initial state for a stable orbit
        
        Parameters:
        -----------
        min_radius : float
            Minimum orbital radius
        max_radius : float
            Maximum orbital radius
        eccentricity_range : tuple
            Range of eccentricity (0 = circular, <1 = elliptical)
        orbit_noise : float
            Small random perturbation to add to the orbit
            
        Returns:
        --------
        initial_state : ndarray
            Initial state [x1, y1, x2, y2, px1, py1, px2, py2]
        """
        # Randomly select orbital parameters
        r = np.random.uniform(min_radius, max_radius)
        eccentricity = np.random.uniform(*eccentricity_range)
        
        # Set initial positions
        # Place bodies along the x-axis, with center of mass at the origin
        x1 = -r * self.m2 / self.total_mass
        y1 = 0.0
        x2 = r * self.m1 / self.total_mass
        y2 = 0.0
        
        # Calculate velocities for a stable orbit
        # For a circular orbit, the velocity is perpendicular to the position
        # and proportional to the square root of G * M / r
        v_orbital = np.sqrt(self.G * self.total_mass / r)
        
        # Adjust for eccentricity (multiply y-velocity component)
        # 1.0 = circular orbit, <1.0 makes it elliptical
        v_factor = np.sqrt(1.0 - eccentricity)
        
        # Velocities in the y-direction
        vy1 = v_orbital * self.m2 / self.total_mass * v_factor
        vy2 = -v_orbital * self.m1 / self.total_mass * v_factor
        
        # Zero velocity in x-direction initially (for simple orbits)
        vx1 = 0.0
        vx2 = 0.0
        
        # Add small random perturbations if noise is specified
        if orbit_noise > 0:
            vx1 += np.random.normal(0, orbit_noise * v_orbital)
            vy1 += np.random.normal(0, orbit_noise * v_orbital)
            vx2 += np.random.normal(0, orbit_noise * v_orbital)
            vy2 += np.random.normal(0, orbit_noise * v_orbital)
        
        # Convert velocities to momenta
        px1 = self.m1 * vx1
        py1 = self.m1 * vy1
        px2 = self.m2 * vx2
        py2 = self.m2 * vy2
        
        return [x1, y1, x2, y2, px1, py1, px2, py2]
    
    def simulate(self, t_span, initial_state=None, num_points=1000,t_eval=None, min_radius=1.0, max_radius=5.0, 
                 eccentricity_range=(0.0, 0.7), orbit_noise=0.0):
        """
        Simulate the system
        
        Parameters:
        -----------
        t_span : tuple
            (t_start, t_end)
        initial_state : ndarray or None
            Initial state [x1, y1, x2, y2, px1, py1, px2, py2]
            If None, a random initial state will be generated
        t_eval : ndarray or None
            Times at which to evaluate the solution
            
        Returns:
        --------
        t : ndarray
            Time points
        y : ndarray
            System states at each time point
        """
        if initial_state is None:
            initial_state = self.generate_initial_state(min_radius, max_radius, eccentricity_range, orbit_noise)

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
        return solution.t, solution.y.T
    
    def calculate_energy(self, state):
        """
        Calculate the total energy of the system (kinetic + potential)
        
        Parameters:
        -----------
        state : ndarray
            System state [x1, y1, x2, y2, px1, py1, px2, py2]
            
        Returns:
        --------
        energy : float
            Total energy (kinetic + potential)
        """
        # Unpack the state vector
        x1, y1, x2, y2, px1, py1, px2, py2 = state
        
        # Calculate the distance between the bodies
        r_vec = np.array([x2 - x1, y2 - y1])
        r = np.sqrt(np.sum(r_vec**2))
        
        # Prevent division by zero
        if r < 1e-10:
            r = 1e-10
        
        # Calculate kinetic energy: T = p^2/(2m)
        T1 = (px1**2 + py1**2) / (2 * self.m1)
        T2 = (px2**2 + py2**2) / (2 * self.m2)
        kinetic = T1 + T2
        
        # Calculate potential energy: U = -G*m1*m2/r
        potential = -self.G * self.m1 * self.m2 / r
        
        # Total energy is kinetic + potential
        return kinetic + potential
    
    def calculate_angular_momentum(self, state):
        """
        Calculate the total angular momentum of the system
        
        Parameters:
        -----------
        state : ndarray
            System state [x1, y1, x2, y2, px1, py1, px2, py2]
            
        Returns:
        --------
        angular_momentum : float
            Total angular momentum
        """
        # Unpack the state vector
        x1, y1, x2, y2, px1, py1, px2, py2 = state
        
        # Calculate angular momentum for each body: L = r × p
        L1 = x1 * py1 - y1 * px1
        L2 = x2 * py2 - y2 * px2
        
        # Total angular momentum
        return L1 + L2


def generate_two_body_dataset(
    save_path='Data/two_body_dataset.pkl',
    num_trajectories=200,
    t_span=(0, 10),
    num_steps=100,
    test_split=0.1,
    G=1.0,
    m1=1.0,
    m2=1.0,
    min_radius=1.0,
    max_radius=5.0,
    eccentricity_range=(0.0, 0.7),
    orbit_noise=0.01,
    random_seed=42,
    num_points=1000
):
    """
    Generate a dataset of two-body gravitational system trajectories
    
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
    G : float
        Gravitational constant
    m1, m2 : float
        Masses of the two bodies
    min_radius, max_radius : float
        Range for initial orbital radius
    eccentricity_range : tuple
        Range of eccentricity (0 = circular, <1 = elliptical)
    orbit_noise : float
        Small random perturbation to add to orbits
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
    
    # Initialize arrays to store state variables
    all_states = np.zeros((num_trajectories, num_steps, 8))  # 8 = state dimension
    all_energies = np.zeros((num_trajectories, num_steps))
    all_angular_momenta = np.zeros((num_trajectories, num_steps))
    
    # Create simulator once with fixed parameters
    simulator = TwoBodySimulator(G=G, m1=m1, m2=m2)
    
    # Generate trajectories with different initial conditions
    for i in tqdm(range(num_trajectories), desc="Generating Two-Body Trajectories"):
        # Generate a random initial state
        initial_state = simulator.generate_initial_state(
            min_radius=min_radius,
            max_radius=max_radius,
            eccentricity_range=eccentricity_range,
            orbit_noise=orbit_noise
        )
        
        # Simulate
        _, states = simulator.simulate(t_span, initial_state, num_points, t_eval)
        
        # Calculate energy and angular momentum at each time point
        energies = np.zeros(num_steps)
        angular_momenta = np.zeros(num_steps)
        
        for j in range(num_steps):
            energies[j] = simulator.calculate_energy(states[j])
            angular_momenta[j] = simulator.calculate_angular_momentum(states[j])
        
        # Check if the simulation is stable (energy doesn't blow up)
        if np.isnan(energies).any() or np.abs(energies).max() > 1e3:
            # Retry with a different initial state
            i -= 1
            continue
        
        # Store in our arrays
        all_states[i] = states
        all_energies[i] = energies
        all_angular_momenta[i] = angular_momenta
    
    # Split into train and test
    num_test = int(num_trajectories * test_split)
    num_train = num_trajectories - num_test
    
    train_states = all_states[:num_train]
    test_states = all_states[num_train:]
    
    train_energies = all_energies[:num_train]
    test_energies = all_energies[num_train:]
    
    train_angular_momenta = all_angular_momenta[:num_train]
    test_angular_momenta = all_angular_momenta[num_train:]
    
    # Create data dictionary
    data_dict = {
        'coords': train_states,
        'test_coords': test_states,
        'energies': train_energies,
        'test_energies': test_energies,
        'angular_momenta': train_angular_momenta,
        'test_angular_momenta': test_angular_momenta,
        'metadata': {
            'G': G,
            'm1': m1,
            'm2': m2,
            't_span': t_span,
            'num_steps': num_steps,
            't_eval': t_eval.tolist(),
            'min_radius': min_radius,
            'max_radius': max_radius,
            'eccentricity_range': eccentricity_range,
            'orbit_noise': orbit_noise,
            'timesteps': num_steps,
            'trials': num_trajectories,
            'nbodies': 2
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as pickle
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    return data_dict


class TwoBodyDataset(torch.utils.data.Dataset):
    def __init__(self, save_path='Data/two_body_dataset.pkl', mode='train', 
                 input_timesteps=4, output_timesteps=1, flatten=False):
        """
        Dataset class for two-body gravitational system trajectories
        
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
            self.states = self.data['coords']
            self.energies = self.data['energies']
            self.angular_momenta = self.data['angular_momenta']
        else:
            self.states = self.data['test_coords']
            self.energies = self.data['test_energies']
            self.angular_momenta = self.data['test_angular_momenta']
        
        # Get data dimensions
        self.feat_dim = self.states.shape[2]  # Should be 8 (x1, y1, x2, y2, px1, py1, px2, py2)
        
        # Prepare input-output sequences
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        
        self.X, self.y = [], []
        self.N = self.states.shape[0]  # Number of trajectories
        trj_timesteps = self.states.shape[1]  # Timesteps per trajectory
        
        # For each trajectory, create sliding windows
        for i in range(self.N):
            for t in range(trj_timesteps - input_timesteps - output_timesteps + 1):
                # Input: sequence of states
                self.X.append(self.states[i, t:t+input_timesteps, :])
                
                # Target: next states
                self.y.append(self.states[i, t+input_timesteps:t+input_timesteps+output_timesteps, :])
        
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
        # Return input states and target states
        return self.X[idx], self.y[idx]


# Functions for visualization
def visualize_orbit(simulator, trajectory, t_values, save_path=None):
    """
    Visualize a two-body orbit
    
    Parameters:
    -----------
    simulator : TwoBodySimulator
        Simulator instance
    trajectory : ndarray
        Trajectory [timesteps, features]
    t_values : ndarray
        Time values
    save_path : str or None
        Path to save the plot, if None, plot is shown
    """
    plt.figure(figsize=(15, 10))
    
    # Plot orbital trajectories
    plt.subplot(2, 3, 1)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', label='Body 1')
    plt.plot(trajectory[:, 2], trajectory[:, 3], 'b-', label='Body 2')
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'ro', label='Body 1 Start')
    plt.plot(trajectory[0, 2], trajectory[0, 3], 'bo', label='Body 2 Start')
    
    # Calculate and plot center of mass
    m1 = simulator.m1
    m2 = simulator.m2
    total_mass = m1 + m2
    com_x = (m1 * trajectory[:, 0] + m2 * trajectory[:, 2]) / total_mass
    com_y = (m1 * trajectory[:, 1] + m2 * trajectory[:, 3]) / total_mass
    plt.plot(com_x, com_y, 'g-', label='Center of Mass')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Orbital Trajectories')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    
    # Plot positions over time
    plt.subplot(2, 3, 2)
    plt.plot(t_values, trajectory[:, 0], 'r-', label='x1')
    plt.plot(t_values, trajectory[:, 1], 'r--', label='y1')
    plt.plot(t_values, trajectory[:, 2], 'b-', label='x2')
    plt.plot(t_values, trajectory[:, 3], 'b--', label='y2')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Positions Over Time')
    plt.grid(True)
    plt.legend()
    
    # Plot momenta over time
    plt.subplot(2, 3, 3)
    plt.plot(t_values, trajectory[:, 4], 'r-', label='px1')
    plt.plot(t_values, trajectory[:, 5], 'r--', label='py1')
    plt.plot(t_values, trajectory[:, 6], 'b-', label='px2')
    plt.plot(t_values, trajectory[:, 7], 'b--', label='py2')
    plt.xlabel('Time')
    plt.ylabel('Momentum')
    plt.title('Momenta Over Time')
    plt.grid(True)
    plt.legend()
    
    # Calculate and plot energy
    plt.subplot(2, 3, 4)
    energies = np.array([simulator.calculate_energy(state) for state in trajectory])
    relative_energy_error = (energies - energies[0]) / np.abs(energies[0])
    plt.plot(t_values, energies, 'g-')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Total Energy Over Time')
    plt.grid(True)
    
    # Plot relative energy error
    plt.subplot(2, 3, 5)
    plt.plot(t_values, relative_energy_error, 'r-')
    plt.xlabel('Time')
    plt.ylabel('(E - E0) / |E0|')
    plt.title('Relative Energy Error')
    plt.grid(True)
    
    # Calculate and plot angular momentum
    plt.subplot(2, 3, 6)
    angular_momenta = np.array([simulator.calculate_angular_momentum(state) for state in trajectory])
    relative_am_error = (angular_momenta - angular_momenta[0]) / np.abs(angular_momenta[0])
    plt.plot(t_values, relative_am_error, 'b-')
    plt.xlabel('Time')
    plt.ylabel('(L - L0) / |L0|')
    plt.title('Relative Angular Momentum Error')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

