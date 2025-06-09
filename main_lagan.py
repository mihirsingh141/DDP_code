import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from Data.spring_mass_dataset import *
from Data.pendulum_dataset import *
from Data.two_body_dataset import *
from Data.lorentz_dataset import *
from src.gan import LieGenerator, LieDiscriminator, LieDiscriminatorEmb
from train import train_lie_gan, train_lie_gan_incremental
import pickle
import os

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")


def get_device():
    """Get the appropriate device for computation"""
    if torch.cuda.is_available():
        return torch.device('cuda:3')
    return torch.device('cpu')
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # model & training settings
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--lr_g', type=float, default=1e-3)
    parser.add_argument('--reg_type', type=str, default='cosine')
    parser.add_argument('--lamda', type=float, default=1e-2)
    parser.add_argument('--p_norm', type=float, default=2)
    parser.add_argument('--droprate_init', type=float, default=0.8)
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--activate_threshold', action='store_true')
    parser.add_argument('--D_loss_threshold', type=float, default=0.25)
    parser.add_argument('--model', type=str, default='lie')
    parser.add_argument('--coef_dist', type=str, default='normal')
    parser.add_argument('--g_init', type=str, default='random')
    parser.add_argument('--sigma_init', type=float, default=1)
    parser.add_argument('--uniform_max', type=float, default=1)
    parser.add_argument('--normalize_Li', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--n_channel', type=int, default=1)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--incremental', action='store_true')
    # dataset settings
    parser.add_argument('--task', type=str, default='spring_mas')
    parser.add_argument('--dataset_name', type=str, default='2body')
    parser.add_argument('--dataset_config', type=str, nargs='+', default=None)
    parser.add_argument('--dataset_size', type=int, default=2000)
    parser.add_argument('--x_type', type=str, default='vector')
    parser.add_argument('--y_type', type=str, default='vector')
    parser.add_argument('--input_timesteps', type=int, default=4)
    parser.add_argument('--output_timesteps', type=int, default=2)
    parser.add_argument('--n_component', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.0)
    # run settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='saved_model')
    parser.add_argument('--save_name', type=str, default='default')
    args = parser.parse_args()
    args.device = get_device()

    os.makedirs(f'{args.save_path}',exist_ok=True)
    with open(f'{args.save_path}/args_{args.task}.pkl','wb') as f:
        pickle.dump(args,f)

    print(args.task)
        
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    if args.task == 'two_body':
        data_dict = generate_two_body_dataset(
            save_path='Data/two_body_dataset.pkl',
            num_trajectories=2000,  # Smaller number for quick testing
            t_span=(0, 10),
            num_steps=100,
            G=1.0,
            m1=1.0,
            m2=1.0,
            min_radius=1.5,
            max_radius=3.0,
            eccentricity_range=(0.0, 0.5),
            orbit_noise=0.01,
            random_seed=42,
            test_split=0.2
        )

        
        dataset = TwoBodyDataset(
            save_path='Data/two_body_dataset.pkl',
            input_timesteps = args.input_timesteps ,
            output_timesteps= args.output_timesteps
        )
        if args.dataset_config is None:
            n_dim = 8
        elif 'log' in args.dataset_config:
            n_dim = 5
        n_channel = args.n_channel
        d_input_size = n_dim * (args.input_timesteps + args.output_timesteps)
    elif args.task == 'spring_mass':
        data_dict = generate_spring_mass_dataset(
            save_path='Data/spring_mass_dataset.pkl',
            num_trajectories=2000,
            t_span=(0, 10),
            num_steps=100,
            test_split=0.2,
            mass=1.0,
            k=1.0,
            damping=0.0,
            random_seed=42
        )
            
        dataset = SpringMassDataset(
            save_path='Data/spring_mass_dataset.pkl',
            input_timesteps = args.input_timesteps,
            output_timesteps = args.output_timesteps
        )
        if args.dataset_config is None:
            n_dim = 2 
        n_channel = args.n_channel
        d_input_size = n_dim * (args.input_timesteps + args.output_timesteps)

    elif args.task == 'pendulum':
        data_dict = generate_pendulum_dataset(
            save_path='Data/pendulum_dataset.pkl',
            num_trajectories=2000,
            t_span=(0, 10),
            num_steps=100,
            test_split=0.2,
            mass = 1.0,
            g=9.8,
            length=1.0,
            damping=0.0,
            random_seed=42
        )


        dataset = PendulumDataset(
            save_path='Data/pendulum_dataset.pkl',
            mode='train',
            input_timesteps= args.input_timesteps,
            output_timesteps = args.output_timesteps
        )
        if args.dataset_config is None:
            n_dim = 2
        n_channel = args.n_channel
        d_input_size = n_dim * (args.input_timesteps + args.output_timesteps)
    elif args.task == 'lorentz':
        data_dict = generate_lorenz_dataset(
            save_path='Data/lorenz_dataset.pkl',
            num_trajectories=2000,
            t_span=(0, 10),  # Longer time span to capture chaotic behavior
            num_steps=100,
            test_split=0.2
        )


        dataset = LorenzDataset(
            save_path='Data/lorenz_dataset.pkl',
            mode='train',
            input_timesteps=args.input_timesteps,
            output_timesteps=args.output_timesteps
        )
        if args.dataset_config is None:
            n_dim=3
        n_channel = args.n_channel
        d_input_size = n_dim * (args.input_timesteps + args.output_timesteps)
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize generator and discriminator
    if args.model in ['lie', 'lie_subgrp']:
        generator = LieGenerator(n_dim, n_channel, args).to(args.device)
    else:
        discriminator = LieDiscriminator(d_input_size).to(args.device)
    if args.model == 'lie':  # fix the coefficient distribution
        generator.mu.requires_grad = False
        generator.sigma.requires_grad = False
    elif args.model == 'lie_subgrp':  # fix the generator
        generator.Li.requires_grad = False

    # Train
    train_fn = train_lie_gan if not args.incremental else train_lie_gan_incremental
    train_fn(
        generator,
        discriminator,
        dataloader,
        args.num_epochs,
        args.lr_d,
        args.lr_g,
        args.reg_type,
        args.lamda,
        args.p_norm,
        args.mu,
        args.eta,
        args.device,
        args.task,
        task=args.task,
        save_path=f'{args.save_path}/LaLieGAN/{args.task}/',
        print_every=args.print_every,
    )
