import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
import math
from collections.abc import Iterable
from tqdm import tqdm, trange
from utils import *
import jax 
import jax.numpy as jnp


save_model_path = 'saved_model'

SEED = 42
random.seed(SEED)
jax.random.PRNGKey(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def get_device():
    """Get the appropriate device for computation"""
    if torch.cuda.is_available():
        return torch.device('cuda:3')
    return torch.device('cpu')
    

# Configure JAX to use GPU if available
if torch.cuda.is_available():
    jax.config.update('jax_platform_name', 'gpu')

def train_lie_gan(
    generator,
    discriminator,
    dataloader,
    num_epochs,
    lr_d,
    lr_g,
    reg_type,
    lamda,
    p_norm,
    mu,
    eta,
    device,
    name,
    task='clf',
    save_path=None,
    print_every=100,
):
    # Loss function
    adversarial_loss = torch.nn.BCELoss(reduction='mean')
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    train_generator = train_discriminator = True

    for ep in trange(num_epochs):
        D_real_loss_list, D_fake_loss_list, G_loss_list, G_reg_list, G_spreg_list, G_chreg_list = [], [], [], [], [], []
        for i, (x, y) in enumerate(dataloader):
            bs = x.shape[0]
            # Adversarial ground truths
            valid = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)
            # Configure input
            x = x.to(device)
            y = y.to(device)
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            # Generate a batch of transformed data points
            gx, gy = generator(x, y)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gx, gy), valid)
            g_spreg = mu * torch.norm(generator.getLi(), p=p_norm)
            if reg_type == 'cosine':
                g_reg = torch.abs(nn.CosineSimilarity(dim=2)(gx, x).mean())
            elif reg_type == 'rel_diff':
                g_reg = -torch.minimum(torch.abs((gx - x) / x).mean(), torch.FloatTensor([1.0]).to(device))
            elif reg_type == 'Li_norm':
                g_reg = -torch.minimum(torch.norm(generator.getLi(), p=2, dim=None), torch.FloatTensor([generator.n_dim * generator.n_channel]).to(device))
            else:
                raise NotImplementedError
            g_reg = lamda * g_reg
            g_chreg = eta * generator.channel_corr(killing=False)
            G_loss_list.append(g_loss.item())
            G_reg_list.append(g_reg.item() / max(lamda, 1e-6))
            G_spreg_list.append(g_spreg.item() / max(mu, 1e-6))
            G_chreg_list.append(g_chreg.item() / max(eta, 1e-6))
            g_loss = g_loss + g_reg + g_spreg + g_chreg
            if train_generator:
                g_loss.backward()
                optimizer_G.step()
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(x, y), valid)
            fake_loss = adversarial_loss(discriminator(gx.detach(), gy.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            if train_discriminator:
                d_loss.backward()
                optimizer_D.step()
            D_real_loss_list.append(real_loss.item())
            D_fake_loss_list.append(fake_loss.item())
        if save_path is not None and (ep + 1) % 100 == 0:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(generator.state_dict(), save_path + f'{name}_generator_{ep}.pt')
            torch.save(discriminator.state_dict(), save_path + f'{name}_discriminator_{ep}.pt')
        if (ep + 1) % print_every == 0:
            print(f'Epoch {ep}: D_real_loss={np.mean(D_real_loss_list)}, D_fake_loss={np.mean(D_fake_loss_list)}, G_loss={np.mean(G_loss_list)}, G_reg={np.mean(G_reg_list)}, G_spreg={np.mean(G_spreg_list)}, G_chreg={np.mean(G_chreg_list)}')
            print(generator.getLi())


def train_lie_gan_incremental(
    generator,
    discriminator,
    dataloader,
    num_epochs,
    lr_d,
    lr_g,
    reg_type,
    lamda,
    p_norm,
    mu,
    eta,
    device,
    task='clf',
    save_path=None,
    print_every=100,
):
    # Loss function
    adversarial_loss = torch.nn.BCELoss(reduction='mean')
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    train_generator = train_discriminator = True

    for ch in range(generator.n_channel):
        generator.set_activated_channel(ch)
        print(f'Training channel {ch}')
        for ep in trange(num_epochs):
            D_real_loss_list, D_fake_loss_list, G_loss_list, G_reg_list, G_spreg_list, G_chreg_list = [], [], [], [], [], []
            for i, (x, y) in enumerate(dataloader):
                # Adversarial ground truths
                valid = torch.ones(x.shape[0], 1, device=device)
                fake = torch.zeros(x.shape[0], 1, device=device)
                # Configure input
                x = x.to(device)
                y = y.to(device)
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                # Generate a batch of transformed data points
                gx, gy = generator(x, y)
                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gx, gy), valid)
                g_spreg = mu * torch.norm(generator.getLi(), p=p_norm)
                if reg_type == 'cosine':
                    g_reg = torch.abs(nn.CosineSimilarity(dim=2)(gx, x).mean())
                elif reg_type == 'rel_diff':
                    g_reg = -torch.minimum(torch.abs((gx - x) / x).mean(), torch.FloatTensor([1.0]).to(device))
                elif reg_type == 'Li_norm':
                    g_reg = -torch.minimum(torch.norm(generator.getLi(), p=2, dim=None), torch.FloatTensor([generator.n_dim * generator.n_channel]).to(device))
                else:
                    raise NotImplementedError
                g_reg = lamda * g_reg
                g_chreg = eta * generator.channel_corr(killing=False)
                G_loss_list.append(g_loss.item())
                G_reg_list.append(g_reg.item() / max(lamda, 1e-6))
                G_spreg_list.append(g_spreg.item() / max(mu, 1e-6))
                G_chreg_list.append(g_chreg.item() / max(eta, 1e-6))
                g_loss = g_loss + g_reg + g_spreg + g_chreg
                if train_generator:
                    g_loss.backward()
                    grad_mask = torch.zeros_like(generator.Li.grad, device=generator.Li.device)
                    grad_mask[ch, :, :] = 1.0
                    generator.Li.grad *= grad_mask  # set other channels to zero
                    optimizer_G.step()
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(x, y), valid)
                fake_loss = adversarial_loss(discriminator(gx.detach(), gy.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                if train_discriminator:
                    d_loss.backward()
                    optimizer_D.step()
                D_real_loss_list.append(real_loss.item())
                D_fake_loss_list.append(fake_loss.item())
            if save_path is not None and (ep + 1) % 100 == 0:
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                torch.save(generator.state_dict(), save_path + f'generator_{ep}.pt')
                torch.save(discriminator.state_dict(), save_path + f'discriminator_{ep}.pt')
            if (ep + 1) % print_every == 0:
                print(f'Epoch {ch}-{ep}: D_real_loss={np.mean(D_real_loss_list)}, D_fake_loss={np.mean(D_fake_loss_list)}, G_loss={np.mean(G_loss_list)}, G_reg={np.mean(G_reg_list)}, G_spreg={np.mean(G_spreg_list)}, G_chreg={np.mean(G_chreg_list)}')
                print(generator.getLi())
        generator.activate_all_channels()


def train_liegerino(
    model,
    dataloader,
    num_epochs,
    lr,
    reg_type,
    lamda,
    p_norm,
    mu,
    eta,
    device,
    task='clf',
    save_path=None,
    print_every=100,
):
    if task == 'top_tagging':
        criterion = nn.CrossEntropyLoss(reduction='mean')
    elif task == 'traj_pred':
        criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = model.aug
    for ep in trange(num_epochs):
        loss_list, reg_list, spreg_list, chreg_list = [], [], [], []
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            gx, fgx, gy = model(x, y)
            loss = criterion(fgx, gy)
            spreg = mu * torch.norm(generator.getLi(), p=p_norm)
            if reg_type == 'cosine':
                reg = torch.abs(nn.CosineSimilarity(dim=2)(gx, x).mean())
            elif reg_type == 'rel_diff':
                reg = torch.abs((gx - x) / x).mean()
            elif reg_type == 'Li_norm':
                reg = -torch.minimum(torch.norm(generator.getLi(), p=2, dim=None), torch.FloatTensor([generator.n_dim * generator.n_channel]).to(device))
            else:
                raise NotImplementedError
            reg = lamda * reg
            chreg = eta * generator.channel_corr(killing=False)
            loss_list.append(loss.item())
            reg_list.append(reg.item() / max(lamda, 1e-6))
            spreg_list.append(spreg.item() / max(mu, 1e-6))
            chreg_list.append(chreg.item() / max(eta, 1e-6))
            loss = loss + reg + spreg + chreg
            loss.backward()
            optimizer.step()
        if save_path is not None and (ep + 1) % 100 == 0:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(model.state_dict(), save_path + f'model_{ep}.pt')
        if (ep + 1) % print_every == 0:
            print(f'Epoch {ep}: loss={np.mean(loss_list)}, reg={np.mean(reg_list)}, spreg={np.mean(spreg_list)}')
            print(generator.getLi())
        

def train_laliegan(
    autoencoder, discriminator, generator, dataloader,
    num_epochs, lr_ae, lr_d, lr_g, w_recon, w_gan, reg_type, w_reg, w_chreg,
    regressor, device, save_interval, save_dir,**kwargs
):
    optimizer_ae = torch.optim.Adam(autoencoder.parameters(),lr=lr_ae) 
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g) 

    adversarial_loss = torch.nn.BCELoss() 
    recon_loss = torch.nn.MSELoss() 

    for epoch in range(num_epochs):
        # torch.autograd.set_detect_anomaly(True)
        running_losses = [[], [], [], [], [], [], [], [], [], []]
        autoencoder.train()
        discriminator.train()
        generator.train()

        for i, (x,y) in enumerate(dataloader):
            x = x.to(device) 
            y = y.to(device) 

            bs = x.shape[0] 

            # Adversarial ground truths
            valid = torch.ones((bs,1)).to(device) 
            fake = torch.zeros((bs,1)).to(device)

            # Reconstruction loss 
            zx,xhat = autoencoder(x) 
            zy,yhat = autoencoder(y)

            loss_ae = w_recon * recon_loss(xhat,x)
            running_losses[0].append(loss_ae.item() / w_recon)
            loss_ae_rel = loss_ae / recon_loss(x, torch.zeros_like(x))
            running_losses[5].append(loss_ae_rel.item() / (w_recon + 1e-6))
            loss = loss_ae 

            # Generator Loss 
            ztx, zty = generator(zx,zy)  
            xt = autoencoder.decode(ztx) 
            xr = xhat 
            d_fake = discriminator(ztx,zty) 
            loss_g = w_gan * adversarial_loss(d_fake, valid)
            running_losses[1].append(loss_g.item() / (w_gan + 1e-6))
            loss = loss + loss_g

            if reg_type == 'Lnorm':
                Ls = generator.getLi()
                loss_g_reg = -sum([torch.minimum(torch.norm(L, p=2, dim=None), torch.FloatTensor([np.prod(L.shape[:-1])]).to(device)) for L in Ls])
                # loss_g_reg = torch.norm(generator.getLi(), p=1, dim=None)
                running_losses[2].append(loss_g_reg.item())
                loss = loss + w_reg * loss_g_reg
            elif reg_type == 'cosine':
                loss_g_reg = torch.abs(nn.CosineSimilarity(dim=-1)(ztx, zx).mean())
                running_losses[2].append(loss_g_reg.item())
                loss = loss + w_reg * loss_g_reg
            elif reg_type == 'none':
                pass
            else:
                raise NotImplementedError
            

            loss_g_chreg = generator.channel_corr()
            running_losses[9].append(loss_g_chreg.item())
            loss = loss + w_chreg * loss_g_chreg 

            # Discriminator Loss 
            zx_detached = zx.detach() 
            ztx_detached = ztx.detach()
            zy_detached = zy.detach()
            zty_detached = zty.detach() 

            xr_detached = xr.detach() 
            xt_detached = xt.detach() 
            loss_d_real = adversarial_loss(discriminator(zx_detached,zy_detached), valid)
            loss_d_fake = adversarial_loss(discriminator(ztx_detached,zty_detached), fake)
            running_losses[3].append(loss_d_real.item())
            running_losses[4].append(loss_d_fake.item())
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss = loss + loss_d

            # Backprop 
            optimizer_ae.zero_grad() 
            optimizer_d.zero_grad() 
            optimizer_g.zero_grad()

            loss.backward() 

            optimizer_ae.step()
            optimizer_d.step() 
            optimizer_g.step()

        
        if (epoch + 1) % save_interval == 0:
            if not os.path.exists(f'{save_model_path}/{save_dir}'):
                os.makedirs(f'{save_model_path}/{save_dir}')
            torch.save(autoencoder.state_dict(), f'{save_model_path}/{save_dir}/laligan_autoencoder_{epoch}.pt')
            torch.save(discriminator.state_dict(), f'{save_model_path}/{save_dir}/laligan_discriminator_{epoch}.pt')
            torch.save(generator.state_dict(), f'{save_model_path}/{save_dir}/laligan_generator_{epoch}.pt')
            torch.save(regressor.state_dict(), f'{save_model_path}/{save_dir}/laligan_regressor_{epoch}.pt')



       

        if epoch%10 == 0 or epoch == num_epochs-1:
            print(f'loss_ae: {np.mean(running_losses[0]):.4f}, loss_g: {np.mean(running_losses[1]):.4f}, loss_g_reg: {np.mean(running_losses[2]):.4f}, loss_g_chreg: {np.mean(running_losses[9]):.4f}, loss_d_real: {np.mean(running_losses[3]):.4f}, loss_d_fake: {np.mean(running_losses[4]):.4f}, loss_ae_rel: {np.mean(running_losses[5]):.4f}')

            


def train_SINDy(
        autoencoder, regressor, train_loader, test_loader,
        num_epochs, lr, reg_type, w_reg, rel_loss,
        device, log_interval, save_interval, save_dir, **kwargs
):
    # Initialize optimizers
    optimizer_sindy = torch.optim.Adam(regressor.parameters(), lr=lr)

    # Loss functions
    sindy_loss = torch.nn.MSELoss()
    recon_loss = torch.nn.MSELoss()

    for epoch in tqdm(range(num_epochs)):
        regressor.train() 
        running_losses = [[], [], [], []] 

        for i, (x,dx) in enumerate(train_loader):
            x = x.to(device)
            dx = dx.to(device) 

            if reg_type == 'l1':
                loss_reg = sum([torch.norm(p, 1) for p in regressor.parameters()])
                running_losses[0].append(loss_reg.item())
            else:
                raise ValueError(f'Unknown regularization type: {reg_type}')
            loss = w_reg * loss_reg


            # Recon loss 
            z,xhat = autoencoder(x) 
            loss_recon = recon_loss(xhat, x)
            running_losses[1].append(loss_recon.item())
            # dz loss & dx loss
            dz = autoencoder.compute_dz(x, dx)
            dz_pred = regressor(z)
            dx_pred = autoencoder.compute_dx(z, dz_pred)

            if rel_loss:
                # Denominator at least 0.1
                denom = torch.max(sindy_loss(dz, torch.zeros_like(dz, device=device)),
                                  torch.ones_like(loss, device=device) * 0.1)
                loss_sindy_z = sindy_loss(dz_pred, dz) / denom
            else:
                loss_sindy_z = sindy_loss(dz_pred, dz)

            loss_sindy_x = sindy_loss(dx_pred, dx)
            running_losses[2].append(loss_sindy_z.item())
            running_losses[3].append(loss_sindy_x.item())
            loss += loss_sindy_z


            # Optimization 
            optimizer_sindy.zero_grad() 
            loss.backward() 
            optimizer_sindy.step() 

        
            if (epoch + 1) % log_interval == 0:
                print(f'Epoch {epoch}, loss_reg: {np.mean(running_losses[0]):.4f}, '
                    f'loss_recon: {np.mean(running_losses[1]):.4f}, '
                    f'loss_sindy_z: {np.mean(running_losses[2]):.4f}, '
                    f'loss_sindy_x: {np.mean(running_losses[3]):.4f}')
                regressor.eval()
                autoencoder.eval()
                with torch.no_grad():
                    running_losses = [[], [], [], []]
                    for i, (x, dx, _) in enumerate(test_loader):
                        x = x.to(device)
                        dx = dx.to(device)
                        z, xhat = autoencoder(x)
                        loss_recon = recon_loss(xhat, x)
                        dz = autoencoder.compute_dz(x, dx)
                        dz_pred = regressor(z)
                        dx_pred = autoencoder.compute_dx(z, dz_pred)
                        loss_sindy_z = sindy_loss(dz_pred, dz)
                        loss_sindy_x = sindy_loss(dx_pred, dx)
                        running_losses[1].append(loss_recon.item())
                        running_losses[2].append(loss_sindy_z.item())
                        running_losses[3].append(loss_sindy_x.item())

                        # Regularization
                        if reg_type == 'l1':
                            loss_reg = sum([torch.norm(p, 1) for p in regressor.parameters()])
                            running_losses[0].append(loss_reg.item())
                        else:
                            raise ValueError(f'Unknown regularization type: {reg_type}')

                    print(f'Epoch {epoch} test, loss_reg: {np.mean(running_losses[0]):.4f}, '
                        f'loss_recon: {np.mean(running_losses[1]):.4f}, '
                        f'loss_sindy_z: {np.mean(running_losses[2]):.4f}, '
                        f'loss_sindy_x: {np.mean(running_losses[3]):.4f}')
                    regressor.print()

            if (epoch + 1) % save_interval == 0:
                        if not os.path.exists(f'{save_model_path}/{save_dir}'):
                            os.makedirs(f'{save_model_path}/{save_dir}')
                        torch.save(regressor.state_dict(), f'{save_model_path}/{save_dir}/regressor_{epoch}.pt')


def train_lassi(
    autoencoder, discriminator, generator, train_loader, test_loader,
    num_epochs, lr_ae, lr_d, lr_g, w_recon, w_gan, reg_type, w_reg, w_chreg,
    regressor, lr_reg, w_reg_z, w_reg_x,
    device, save_interval, save_dir,**kwargs
):
    def validate(regressor,test_loader,autoencoder):
        regressor.eval()
        autoencoder.eval() 

        val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for x,y in test_loader:
                x,y = x.to(device),y.to(device)
                
                try:
                    zx,_ = autoencoder(x) 
                    zy,_ = autoencoder(y)

                    pred_zy = regressor(zx) 
                    loss_reg_z = regressor_loss(pred_zy,zy) 
                    pred_y = autoencoder.decode(pred_zy)
                    loss_reg_x = regressor_loss(pred_y,y) 

                    val_loss += loss_reg_x.item() 
                    num_batches += 1
                
                except Exception as e:
                    print(f'Error in validation batch: {e}') 
                    continue 

        regressor.train()
        autoencoder.train()

        if num_batches == 0:
            return float('inf') 
        else:
            return val_loss/num_batches 
    

    optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=lr_ae) 
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g)
    optimizer_reg = torch.optim.Adam(regressor.parameters(), lr=lr_reg)
    # scheduler_reg = torch.optim.lr_scheduler.MultiStepLR(optimizer_reg, milestones=[1, 2, 3], gamma=10)
    regressor_loss = torch.nn.MSELoss()
    adversarial_loss = torch.nn.BCELoss()
    recon_loss = torch.nn.MSELoss()

    stats = {
        'train_loss':[],
        'test_loss':[],
        'grad_norm':[],
    }

    print(f"Starting training: {num_epochs} epochs")

    for epoch in tqdm(range(num_epochs)):
        running_losses = [[], [], [], [], [], [], [], [], [], []]
        autoencoder.train()
        discriminator.train()
        generator.train()
        regressor.train()

        train_loss = 0
        epoch_grad_norm = 0
        num_batches = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device) 
            y = y.to(device)
            bs = x.shape[0] 

            # Adversarial ground truths
            valid = torch.ones((bs, 1)).to(device)
            fake = torch.zeros((bs, 1)).to(device)

            # Reconstruction loss 
            zx,xhat = autoencoder(x) 
            zy,yhat = autoencoder(y)

            loss_ae = w_recon * (recon_loss(xhat,x) + recon_loss(yhat,y))
            running_losses[0].append(loss_ae.item() / w_recon)
            loss_ae_rel = loss_ae / (recon_loss(x, torch.zeros_like(x)) + recon_loss(y,torch.zeros_like(y)))
            running_losses[5].append(loss_ae_rel.item() / (w_recon + 1e-6))
            loss = loss_ae 

            # Generator Loss
            ztx, zty = generator(zx,zy)  
            xt = autoencoder.decode(ztx) 
            xr = xhat 
            d_fake = discriminator(ztx,zty) 
            loss_g = w_gan * adversarial_loss(d_fake, valid)
            running_losses[1].append(loss_g.item() / (w_gan + 1e-6))
            loss = loss + loss_g

            if reg_type == 'Lnorm':
                Ls = generator.getLi()
                loss_g_reg = -sum([torch.minimum(torch.norm(L, p=2, dim=None), torch.FloatTensor([np.prod(L.shape[:-1])]).to(device)) for L in Ls])
                # loss_g_reg = torch.norm(generator.getLi(), p=1, dim=None)
                running_losses[2].append(loss_g_reg.item())
                loss = loss + w_reg * loss_g_reg
            elif reg_type == 'cosine':
                loss_g_reg = torch.abs(nn.CosineSimilarity(dim=-1)(ztx, zx).mean())
                running_losses[2].append(loss_g_reg.item())
                loss = loss + w_reg * loss_g_reg
            elif reg_type == 'none':
                pass
            else:
                raise NotImplementedError  
            
            loss_g_chreg = generator.channel_corr()
            running_losses[9].append(loss_g_chreg.item())
            loss = loss + w_chreg * loss_g_chreg 


            # Discriminator Loss 
            zx_detached = zx.detach() 
            ztx_detached = ztx.detach()
            zy_detached = zy.detach()
            zty_detached = zty.detach() 

            xr_detached = xr.detach() 
            xt_detached = xt.detach() 
            loss_d_real = adversarial_loss(discriminator(zx_detached,zy_detached), valid)
            loss_d_fake = adversarial_loss(discriminator(ztx_detached,zty_detached), fake)
            running_losses[3].append(loss_d_real.item())
            running_losses[4].append(loss_d_fake.item())
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss = loss + loss_d

            # Regressor loss 
            try:
                pred_zy = regressor(zx) 
                loss_reg_z = regressor_loss(pred_zy,zy) 
                pred_y = autoencoder.decode(pred_zy)
                loss_reg_x = regressor_loss(pred_y,y) 

                loss = loss + w_reg_z*loss_reg_z + w_reg_x*loss_reg_x

                grad_norm =  0.0 
                for p in regressor.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                epoch_grad_norm += grad_norm

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(regressor.parameters(), max_norm=1.0) 

                train_loss += loss_reg_x.item()
                num_batches += 1

                    
                # Backprop 
                optimizer_ae.zero_grad() 
                optimizer_d.zero_grad()
                optimizer_g.zero_grad()
                optimizer_reg.zero_grad()
                loss.backward()

                optimizer_ae.step() 
                optimizer_d.step()
                optimizer_g.step()
                optimizer_reg.step()

            except Exception as e:
                print(f'Error in training batch: {e}')
            
        if num_batches == 0:
            print('No valid batches in training epoch')
            continue

        # Calculate epoch-level metrics
        train_loss /= num_batches
        epoch_grad_norm /= num_batches
        stats['train_loss'].append(train_loss)
        stats['grad_norm'].append(epoch_grad_norm)

        # Run full validation on test set
        test_loss = validate(regressor,test_loader,autoencoder)
        stats['test_loss'].append(test_loss)

        

        if epoch%10 == 0 or epoch == num_epochs-1:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            print(f'loss_ae: {np.mean(running_losses[0]):.4f}, loss_g: {np.mean(running_losses[1]):.4f}, loss_g_reg: {np.mean(running_losses[2]):.4f}, loss_g_chreg: {np.mean(running_losses[9]):.4f}, loss_d_real: {np.mean(running_losses[3]):.4f}, loss_d_fake: {np.mean(running_losses[4]):.4f}, loss_ae_rel: {np.mean(running_losses[5]):.4f}')
            print(generator.getLi())

        if (epoch + 1) % save_interval == 0:
            if not os.path.exists(f'{save_model_path}/{save_dir}'):
                os.makedirs(f'{save_model_path}/{save_dir}')
            torch.save(autoencoder.state_dict(), f'{save_model_path}/{save_dir}/laligan_autoencoder_{epoch}.pt')
            torch.save(discriminator.state_dict(), f'{save_model_path}/{save_dir}/laligan_discriminator_{epoch}.pt')
            torch.save(generator.state_dict(), f'{save_model_path}/{save_dir}/laligan_generator_{epoch}.pt')
            torch.save(regressor.state_dict(), f'{save_model_path}/{save_dir}/laligan_regressor_{epoch}.pt')


    return stats


