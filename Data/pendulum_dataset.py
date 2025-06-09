# import numpy as np
# from scipy.integrate import odeint
# from tqdm import trange 
# from torch.utils.data import Dataset 
# import torch 


# def get_pendulum_data(n_ics):
#     t, x, dx, ddx, z = generate_pendulum_data(n_ics)
#     data = {}
#     data['t'] = t
#     data['x'] = x.reshape((n_ics*t.size, -1))
#     data['dx'] = dx.reshape((n_ics*t.size, -1))
#     data['ddx'] = ddx.reshape((n_ics*t.size, -1))
#     data['z'] = z.reshape((n_ics*t.size, -1))[:, 0:1]
#     data['dz'] = z.reshape((n_ics*t.size, -1))[:, 1:2]

#     return data


# def get_low_dim_pendulum_data(n_ics):
#     t, z, dz = generate_low_dim_pendulum_data(n_ics)
#     data = {}
#     data['t'] = t
#     data['z'] = z.reshape((n_ics*t.size, -1))
#     data['dz'] = dz.reshape((n_ics*t.size, -1))

#     return data


# def generate_pendulum_data(n_ics):
#     f = lambda z, t: [z[1], -np.sin(z[0])]
#     t = np.arange(0, 10, .02)

#     z = np.zeros((n_ics, t.size, 2))
#     dz = np.zeros(z.shape)

#     z1range = np.array([-np.pi, np.pi])
#     z2range = np.array([-2.1, 2.1])
#     i = 0
#     for i in trange(n_ics):
#         z0 = np.array([(z1range[1]-z1range[0])*np.random.rand()+z1range[0],
#                        (z2range[1]-z2range[0])*np.random.rand()+z2range[0]])
#         while np.abs(z0[1]**2/2. - np.cos(z0[0])) > .99:
#             z0 = np.array([(z1range[1]-z1range[0])*np.random.rand()+z1range[0],
#                            (z2range[1]-z2range[0])*np.random.rand()+z2range[0]])
#         z[i] = odeint(f, z0, t)
#         dz[i] = np.array([f(z[i, j], t[j]) for j in range(len(t))])

#     print('Converting to movie...')
#     x, dx, ddx = pendulum_to_movie(z, dz)

#     return t, x, dx, ddx, z


# def generate_low_dim_pendulum_data(n_ics):
#     f = lambda z, t: [z[1], -np.sin(z[0])]
#     t = np.arange(0, 10, .02)

#     z = np.zeros((n_ics, t.size, 2))
#     dz = np.zeros(z.shape)

#     z1range = np.array([-np.pi, np.pi])
#     z2range = np.array([-2.1, 2.1])
#     i = 0
#     for i in trange(n_ics):
#         z0 = np.array([(z1range[1]-z1range[0])*np.random.rand()+z1range[0],
#                        (z2range[1]-z2range[0])*np.random.rand()+z2range[0]])
#         while np.abs(z0[1]**2/2. - np.cos(z0[0])) > .99:
#             z0 = np.array([(z1range[1]-z1range[0])*np.random.rand()+z1range[0],
#                            (z2range[1]-z2range[0])*np.random.rand()+z2range[0]])
#         z[i] = odeint(f, z0, t)
#         dz[i] = np.array([f(z[i, j], t[j]) for j in range(len(t))])

#     return t, z, dz


# def H_pendulum(x):
#     return 0.5*x[:, 1]**2 + 1 - np.cos(x[:, 0])


# def pendulum_to_movie(z, dz):
#     n_ics = z.shape[0]
#     n_samples = z.shape[1]
#     n = 51
#     y1, y2 = np.meshgrid(np.linspace(-1.5, 1.5, n), np.linspace(1.5, -1.5, n))
#     create_image = lambda theta: np.exp(-((y1-np.cos(theta-np.pi/2))**2 + (y2-np.sin(theta-np.pi/2))**2)/.05)
#     argument_derivative = lambda theta, dtheta: \
#         -1/.05*(2*(y1 - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta
#                 + 2*(y2 - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta)
#     argument_derivative2 = lambda theta, dtheta, ddtheta: \
#         -2/.05*((np.sin(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta**2
#                 + (y1 - np.cos(theta-np.pi/2))*np.cos(theta-np.pi/2)*dtheta**2
#                 + (y1 - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*ddtheta
#                 + (-np.cos(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta**2
#                 + (y2 - np.sin(theta-np.pi/2))*(np.sin(theta-np.pi/2))*dtheta**2
#                 + (y2 - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*ddtheta)

#     x = np.zeros((n_ics, n_samples, n, n))
#     dx = np.zeros((n_ics, n_samples, n, n))
#     ddx = np.zeros((n_ics, n_samples, n, n))
#     for i in trange(n_ics):
#         for j in range(n_samples):
#             z[i, j, 0] = wrap_to_pi(z[i, j, 0])
#             x[i, j] = create_image(z[i, j, 0])
#             dx[i, j] = (create_image(z[i, j, 0])*argument_derivative(z[i, j, 0], dz[i, j, 0]))
#             ddx[i, j] = create_image(z[i, j, 0])*((argument_derivative(z[i, j, 0], dz[i, j, 0]))**2
#                                                   + argument_derivative2(z[i, j, 0], dz[i, j, 0], dz[i, j, 1]))
 
#     return x, dx, ddx


# def wrap_to_pi(z):
#     z_mod = z % (2*np.pi)
#     subtract_m = (z_mod > np.pi) * (-2*np.pi)
#     return z_mod + subtract_m



# class PendulumDataset(Dataset): 
#     def __init__(self, path=f'pendulum', n_timesteps=2, mode='train'):
#         super().__init__()
#         try:
#             print(f'Loading existing pendulum {mode} data...')
#             x = torch.load(f'{path}/{mode}-x.pt')
#             dx = torch.load(f'{path}/{mode}-dx.pt')
#             ddx = torch.load(f'{path}/{mode}-ddx.pt')
#         except FileNotFoundError:
#             print(f'Load data failed. Generating pendulum {mode} data...')
#             n_ics = 200 if mode == 'train' else 20
#             data = get_pendulum_data(n_ics=n_ics)
#             x = data['x'].reshape(n_ics, -1, data['x'].shape[-1])
#             dx = data['dx'].reshape(n_ics, -1, data['dx'].shape[-1])
#             ddx = data['ddx'].reshape(n_ics, -1, data['ddx'].shape[-1])
#             x = torch.FloatTensor(x)
#             dx = torch.FloatTensor(dx)
#             ddx = torch.FloatTensor(ddx)
#             torch.save(x, f'{path}/{mode}-x.pt')
#             torch.save(dx, f'{path}/{mode}-dx.pt')
#             torch.save(ddx, f'{path}/{mode}-ddx.pt')
#         self.n_timesteps = n_timesteps
#         n_ics, n_steps, input_dim = x.shape
#         self.n_ics, self.n_steps, self.input_dim = n_ics, n_steps, input_dim
#         self.x = []
#         self.dx = []
#         self.ddx = []
#         for i in range(n_ics):
#             for j in range(n_steps-n_timesteps):
#                 self.x.append(x[i, j:j+n_timesteps, :].reshape((n_timesteps, input_dim)))
#                 self.dx.append(dx[i, j:j+n_timesteps, :].reshape((n_timesteps, input_dim)))
#                 self.ddx.append(ddx[i, j:j+n_timesteps, :].reshape((n_timesteps, input_dim)))

#         self.x = torch.stack(self.x)
#         self.dx = torch.stack(self.dx)
#         self.ddx = torch.stack(self.ddx)

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         return torch.cat([self.x[idx], self.dx[idx]], dim=-1), torch.cat([self.dx[idx], self.ddx[idx]], dim=-1), torch.cat([self.dx[idx], self.ddx[idx]], dim=-1) 



import matplotlib.pyplot as plt
import pickle
import torch
import os
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


class PendulumSimulator:
    def __init__(self, mass=1.0, g=9.8, length=1.0, damping=0.0):
        """
        Initialize a simple pendulum system simulator
        
        Parameters:
        -----------
        g : float
            Gravitational acceleration
        length : float
            Length of the pendulum
        damping : float
            Damping coefficient
        """
        self.g = g
        self.length = length
        self.damping = damping
        self.mass = mass 
    
    def derivatives(self, t, state):
        """
        Calculate derivatives for the system (dθ/dt = ω, dω/dt = α)
        
        Parameters:
        -----------
        t : float
            Time (not used in time-invariant systems)
        state : ndarray
            System state [theta, omega]
            
        Returns:
        --------
        derivatives : ndarray
            Derivatives [omega, alpha]
        """
        theta, omega = state
        
        # Pendulum dynamics: d²θ/dt² = -g/L * sin(θ) - c*dθ/dt
        alpha = -self.g/self.length * np.sin(theta) - self.damping * omega
        
        return [omega, alpha]
    
    def simulate(self, t_span, initial_state=None, num_points=1000,t_eval=None):
        """
        Simulate the system
        
        Parameters:
        -----------
        t_span : tuple
            (t_start, t_end)
        initial_state : ndarray or None
            Initial state [theta, omega]
            If None, random initial theta between -π and π and zero angular velocity
        t_eval : ndarray or None
            Times at which to evaluate the solution
            
        Returns:
        --------
        t : ndarray
            Time points
        y : ndarray
            System states at each time point [thetas, omegas]
        """
        if initial_state is None:
            # Random initial angle between -π and π
            initial_theta = np.random.uniform(-np.pi, np.pi)
            # Zero initial angular velocity
            initial_omega = 0.0
            initial_state = [initial_theta, initial_omega]

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
        
        # Wrap angles to [-π, π]
        thetas = solution.y[0]
        # wrapped_thetas = self.wrap_to_pi(thetas)
        
        # Replace with wrapped angles
        # solution.y[0] = wrapped_thetas
        
        # Reshape to [timesteps, features]
        return solution.t, np.array([solution.y[0], solution.y[1]]).T
    
    @staticmethod
    def wrap_to_pi(theta):
        """
        Wrap angle to [-π, π]
        
        Parameters:
        -----------
        theta : float or ndarray
            Angle(s) to wrap
            
        Returns:
        --------
        wrapped_theta : float or ndarray
            Wrapped angle(s)
        """
        theta_mod = theta % (2*np.pi)
        subtract_m = (theta_mod > np.pi) * (-2*np.pi)
        return theta_mod + subtract_m
    
    def calculate_energy(self, state):
        """
        Calculate the total energy of the pendulum
        
        Parameters:
        -----------
        state : ndarray
            System state [theta, omega]
            
        Returns:
        --------
        energy : float
            Total energy (kinetic + potential)
        """
        theta, omega = state
        
        # Kinetic energy: K = 0.5 * m * L² * ω²
        # For unit mass (m=1), K = 0.5 * L² * ω²
        kinetic = 0.5 * self.mass * self.length**2 * omega**2
        
        # Potential energy: U = m*g*L*(1-cos(θ))
        # For unit mass (m=1), U = g*L*(1-cos(θ))
        potential = self.mass * self.g * self.length * (1 - np.cos(theta))
        
        return kinetic + potential


def generate_pendulum_dataset(
    save_path='Data/pendulum_dataset.pkl',
    num_trajectories=200,
    t_span=(0, 10),
    num_steps=100,
    test_split=0.1,
    mass = 1.0,
    g=9.8,
    length=1.0,
    damping=0.0,
    random_seed=42,
    num_points=1000
):
    """
    Generate a dataset of simple pendulum system trajectories
    
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
    g : float
        Gravitational acceleration
    length : float
        Length of the pendulum
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
    
    # Initialize arrays to store state variables (theta and omega)
    all_states = np.zeros((num_trajectories, num_steps, 2))
    # all_derivatives = np.zeros((num_trajectories, num_steps, 2))
    all_energies = np.zeros((num_trajectories, num_steps))
    
    # Create simulator once with fixed parameters
    simulator = PendulumSimulator(g=g, length=length, damping=damping)
    
    # Generate trajectories with different initial conditions
    for i in tqdm(range(num_trajectories), desc="Generating Pendulum Trajectories"):
        # Generate a random initial state
        initial_theta = np.random.uniform(-np.pi, np.pi)
        initial_omega = np.random.uniform(-2.0, 2.0)
        
        # Avoid starting near the edge of phase space where energy is high
        energy = simulator.calculate_energy([initial_theta, initial_omega])
        energy_threshold = 2.0 * g * length  # Adjust threshold based on system
        
        # Retry until we get initial conditions with reasonable energy
        while energy > energy_threshold:
            initial_theta = np.random.uniform(-np.pi, np.pi)
            initial_omega = np.random.uniform(-2.0, 2.0)
            energy = simulator.calculate_energy([initial_theta, initial_omega])
        
        initial_state = [initial_theta, initial_omega]
        
        # Simulate
        _, states = simulator.simulate(t_span, initial_state, num_points, t_eval)
        
        # Calculate derivatives at each time point
        # derivatives = np.zeros_like(states)
        # for j in range(num_steps):
        #     derivatives[j] = simulator.derivatives(t_eval[j], states[j])
        
        # Calculate energy at each time point
        energies = np.zeros(num_steps)
        for j in range(num_steps):
            energies[j] = simulator.calculate_energy(states[j])
        
        # Store in our arrays
        all_states[i] = states
        # all_derivatives[i] = derivatives
        all_energies[i] = energies
    
    # Split into train and test
    num_test = int(num_trajectories * test_split)
    num_train = num_trajectories - num_test
    
    train_states = all_states[:num_train]
    test_states = all_states[num_train:]
    
    # train_derivatives = all_derivatives[:num_train]
    # test_derivatives = all_derivatives[num_train:]
    
    train_energies = all_energies[:num_train]
    test_energies = all_energies[num_train:]
    
    # Create data dictionary
    data_dict = {
        'states': train_states,
        'test_states': test_states,
        # 'derivatives': train_derivatives,
        # 'test_derivatives': test_derivatives,
        'energies': train_energies,
        'test_energies': test_energies,
        'metadata': {
            'mass':mass,
            'g': g,
            'length': length,
            'damping': damping,
            't_span': t_span,
            'num_steps': num_steps,
            't_eval': t_eval.tolist()
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as pickle
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    return data_dict


class PendulumDataset(torch.utils.data.Dataset):
    def __init__(self, save_path='Data/pendulum_dataset.pkl', mode='train', 
                 input_timesteps=4, output_timesteps=1, flatten=False):
        """
        Dataset class for pendulum system trajectories
        
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
            self.states = self.data['states']
            # self.derivatives = self.data['derivatives']
            self.energies = self.data['energies']
        else:
            self.states = self.data['test_states']
            # self.derivatives = self.data['test_derivatives']
            self.energies = self.data['test_energies']
        
        # Get data dimensions
        self.feat_dim = self.states.shape[2]  # Should be 2 (theta, omega)
        
        # Prepare input-output sequences
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        
        # self.X, self.y, self.dX, self.ddy = [], [], [], []
        self.X, self.y = [], []
        self.N = self.states.shape[0]  # Number of trajectories
        trj_timesteps = self.states.shape[1]  # Timesteps per trajectory
        
        # For each trajectory, create sliding windows
        for i in range(self.N):
            for t in range(trj_timesteps - input_timesteps - output_timesteps):
                # Input: sequence of states
                self.X.append(self.states[i, t:t+input_timesteps, :])
                
                # Target: next states
                self.y.append(self.states[i, t+input_timesteps:t+input_timesteps+output_timesteps, :])
                
                # Derivatives: sequence of derivatives
                # self.dX.append(self.derivatives[i, t:t+input_timesteps, :])
                
                # Second derivatives: next derivatives
                # self.ddy.append(self.derivatives[i, t+input_timesteps:t+input_timesteps+output_timesteps, :])
        
        # Convert to PyTorch tensors
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
        # self.dX = torch.tensor(np.array(self.dX), dtype=torch.float32)
        # self.ddy = torch.tensor(np.array(self.ddy), dtype=torch.float32)
        
        # Get dataset length
        self.len = self.X.shape[0]
        
        # Flatten if requested
        if flatten:
            self.X = self.X.reshape(self.len, -1)
            self.y = self.y.reshape(self.len, -1)
            # self.dX = self.dX.reshape(self.len, -1)
            # self.ddy = self.ddy.reshape(self.len, -1)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # Return states, derivatives, and targets
        # Format: (input_states, target_states, input_derivatives, target_derivatives)
        return self.X[idx], self.y[idx]


# Example usage for visualization
def visualize_pendulum_trajectory(simulator, trajectory, t_values, save_path=None):
    """
    Visualize a pendulum trajectory
    
    Parameters:
    -----------
    simulator : PendulumSimulator
        Simulator instance
    trajectory : ndarray
        Trajectory [timesteps, features]
    t_values : ndarray
        Time values
    save_path : str or None
        Path to save the plot, if None, plot is shown
    """
    plt.figure(figsize=(12, 8))
    
    # Plot theta
    plt.subplot(2, 2, 1)
    plt.plot(t_values, trajectory[:, 0])
    plt.xlabel('Time')
    plt.ylabel('Angle (rad)')
    plt.title('Pendulum Angle Over Time')
    plt.grid(True)
    
    # Plot omega
    plt.subplot(2, 2, 2)
    plt.plot(t_values, trajectory[:, 1])
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity Over Time')
    plt.grid(True)
    
    # Plot phase space
    plt.subplot(2, 2, 3)
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.xlabel('Angle (rad)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Phase Space')
    plt.grid(True)
    
    # Calculate and plot energy
    plt.subplot(2, 2, 4)
    energies = np.array([simulator.calculate_energy(state) for state in trajectory])
    plt.plot(t_values, energies)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Total Energy Over Time')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


