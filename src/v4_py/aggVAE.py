
import os 
import jax 
from jax import jit , lax 
import jax.numpy as jnp 
import jax.random as random
from jax.random import PRNGKey
from jaxtyping import Float, Array
from typing import Union
import numpyro 
import numpyro.distributions as dist 
from numpyro.contrib.module import flax_module
from numpyro.infer import Predictive, SVI, RenyiELBO

import flax.linen as nn
import torch 
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np
import geopandas as gpd

from bokeh.plotting import figure, show
from bokeh.io import export_png

import argparse
import pickle
from functools import partial
from termcolor import colored

from construct_spatial_matrix import compute_pts_in_polys
from agg_gp_priors import exp_sq_kernel, M_g

# ------------------------------ Aggregated GPs ------------------------------ #
def gp_aggr(
    x:Float[Array, "n_pts 2"],
    M_lo:Float[Array, "n_reg n_pts"],
    M_hi:Float[Array, "n_reg n_pts"],
    popw,
    noise:float = 1e-4,
    jitter:float = 1e-4,
):
    # GP hyperparams
    kernel_length = numpyro.sample("kernel_length",dist.InverseGamma(3,3))
    kernel_var = numpyro.sample("kernel_var", dist.HalfNormal(0.05))
    # GP kernel 
    k = exp_sq_kernel(x,x,kernel_var,kernel_length, noise,jitter) # (grd_pts, grd_pts)
    # Sample a GP 
    f = numpyro.sample(
        "f",
        dist.MultivariateNormal(
            loc = jnp.zeros(x.shape[0]),
            covariance_matrix = k,
        ),
        obs = None
    ) # (grid_pts,)
    # Aggregated Gps for all points in regions (low/high)
    gp_aggr_lo = numpyro.deterministic(
        "gp_aggr_lo",
        M_g(M_lo,f,popw)
    ) # (n_regions_lo,)
    gp_aggr_hi = numpyro.deterministic(
        "gp_aggr_hi",
        M_g(M_hi,f,popw),
    ) # (n_regions_hi,)
    gp_aggr = numpyro.deterministic(
        "gp_aggr",
        jnp.concatenate([gp_aggr_lo, gp_aggr_hi], axis = 0)
    ) # (n_regions_lo+ho,)
    return gp_aggr #(56,)

# ------------------------------------ VAE ----------------------------------- #
class Encoder(nn.Module):
    """VAE Encoder"""
    hidden_dim: int = 50  # hidden dimension
    z_dim: int = 40 # latent dimension

    @nn.compact
    def __call__(self,x):
        h_out = nn.elu(nn.Dense(self.hidden_dim)(x))
        z_loc = nn.Dense(self.z_dim)(h_out)
        #z_scale = jnp.exp(nn.Dense(self.z_dim)(h_out)) + 1e-5
        z_scale = nn.softplus(nn.Dense(self.z_dim)(h_out)) + 1e-5
        return z_loc, z_scale

class Decoder(nn.Module):
    """VAE Decoder"""
    hidden_dim: int = 50 # Hidden dimension
    out_dim: int = 56 # Number of regions (9 + 47)

    @nn.compact 
    def __call__(self, z):
        h_out = nn.elu(nn.Dense(self.hidden_dim)(z)) #(*,40) -> (*,50)
        x_out = nn.Dense(self.out_dim)(h_out) #(*,50) -> (*,56)
        return x_out

def guide(batch, hidden_dim = 50, z_dim = 40, beta = None):
    """ This is the encoder part of the VAE"""
    n_samples, input_dim = batch.shape # i.e (*,56)
    encode = numpyro.contrib.module.flax_module(
        name = "encoder",
        nn_module = Encoder(hidden_dim, z_dim), 
        #input_shape = (n_samples, input_dim)
        input_shape=(1, input_dim)
    )
    z_loc, z_std = encode(batch)

    # beta VAE
    if beta:
        with numpyro.plate("batch", n_samples):
            with numpyro.handlers.scale(scale = beta):
                return numpyro.sample(
                    "z",
                    dist.Normal(z_loc,z_std).to_event(1)
                )
    # regular VAE
    else:
        # Note n_samples here is the batch_size (i.e 64)
        # Usage of "plate" is to ensure the latent vectors (z)
        # are indpendent.
        with numpyro.plate("batch", n_samples): 
            # to_event treats 40 (because z is 40) indpendent 
            # distributions as one vector containing 40 distributions
            return numpyro.sample(
                "z",
                dist.Normal(z_loc,z_std).to_event(1)
            )


def model(batch, hidden_dim = 50, z_dim = 40, beta = None):
    """This is decoder part of the VAE"""
    n_samples , out_dim = batch.shape #i.e (*,56)
    decode = numpyro.contrib.module.flax_module(
        name = "decoder",
        nn_module = Decoder(
            hidden_dim = hidden_dim, 
            out_dim = out_dim),
        #input_shape = (n_samples, z_dim)
        input_shape = (1,z_dim)
    )
    if beta:
        with numpyro.plate("batch", n_samples):
            with numpyro.handlers.scale(scale = beta):
                z = numpyro.sample(
                    "z",
                    dist.Normal(
                        jnp.zeros((n_samples, z_dim)),
                        jnp.ones((n_samples, z_dim))
                    ).to_event(1)
                )
                regions_mu = decode(z)
                return numpyro.sample(
                    "obs",
                    dist.Normal(regions_mu, 1).to_event(1),
                    obs = batch
                )
    else:
        # Generate a simple distribution
        with numpyro.plate("batch", n_samples):
            z = numpyro.sample(
                "z", 
                dist.Normal(
                    jnp.zeros((n_samples, z_dim)),
                    jnp.ones((n_samples, z_dim))
                ).to_event(1)
            )

            regions_mu = decode(z) #(*, 56)
            return numpyro.sample(
                "obs",
                dist.Normal(regions_mu, 1).to_event(1),
                obs = batch
            )

# ------------------------------ Training Funcs ------------------------------ #

def jax_collate(batch):
    """Converts a torch tensor to a Jax array"""
    return jax.tree_util.tree_map(
        jnp.array, 
        torch.utils.data.default_collate(batch)
    )

def epoch_train_fixed_gps(svi_state, trainloader):
    """Train model on fixed dataset (pytorch dataloader)"""
    loss_sum = 0.0
    for i, (gp_batch,) in enumerate(trainloader):  # note: (gp_batch,) because of TensorDataset
        #rng_key_i = jax.random.fold_in(rng_key, i)
        svi_state, loss = svi.update(svi_state, gp_batch)
        num_samples = gp_batch.shape[0]
        loss_sum += loss / num_samples
    return svi_state, loss_sum / len(trainloader)

def epoch_eval_fixed_gps(svi_state, valloader):
    """Evaluate model on fixed dataset (pytorch dataloader)"""
    loss_sum = 0.0 
    for i, (gp_batch,) in enumerate(valloader):
        num_samples = gp_batch.shape[0]
        # svi_state is returned by evaluate but we dont need it
        loss = svi.evaluate(svi_state, gp_batch)
        loss_sum += loss / num_samples
    return loss_sum / len(valloader)

@jit
def epoch_train_gen_gp_on_fly(
    svi_state, rng_key, num_samples, num_batches, 
    x, M_lo, M_hi, popw
):
    """Generate gp realizations at train time, every epoch see a new set
    of gps""" 
    def body_fn(i, val):
        # get total loss and svi state 
        loss_sum, svi_state = val 
        # we use random.fol_in method to generate keys
        rng_key_i = jax.random.fold_in(rng_key, i)
        # Everytime we instantiate a new set of GP's 
        gp_samples = gp_aggr_predictive(
            rng_key_i,
            x = x, 
            M_lo = M_lo, 
            M_hi = M_hi,
            popw = popw, 
            noise = 1e-4,
            jitter = 1e-4
        )["gp_aggr"] #(*,56)
        # Get svit state and loss 
        svi_state, loss = svi.update(svi_state, gp_samples)
        # accumulate per sample loss 
        loss_sum += loss / num_samples 
        
        return loss_sum, svi_state 
    
    loss, svi_state = lax.fori_loop(
        lower = 0,
        upper = num_batches,
        body_fun = body_fn,
        init_val = (0.0, svi_state)
    )
    return loss / num_batches, svi_state

@jit
def epoch_eval_gen_gp_on_fly(svi_state, rng_key, num_samples, num_batches, x, M_lo, M_hi,popw):
    """Evaluate model on GP's generated on fly, same gp is not seen by the model in the following epochs"""
    def body_fn(i, loss_sum):
        # use random.fold_in to generate keys 
        rng_key_i = jax.random.fold_in(rng_key, i)
        # Generate GPs
        
        gp_samples = gp_aggr_predictive(
            rng_key_i,
            x = x,
            M_lo = M_lo, 
            M_hi = M_hi, 
            popw = popw, 
            jitter = 1e-4,
            noise = 1e-4
        )["gp_aggr"]
        
        loss = svi.evaluate(svi_state, gp_samples)
        loss_sum += loss / num_samples 
        return loss_sum 

    loss = lax.fori_loop(
        lower = 0, 
        upper = num_batches, 
        body_fun = body_fn, 
        init_val = 0.0
    )
    return loss / num_batches
# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Run VAE model")
    parser.add_argument("--data_root", type = str, default = "data/processed/v4_clin_labs")
    parser.add_argument("--n_samples", type = int, default = 32, help = "number of samples")
    parser.add_argument("--n_batches", type = int, default = 10, help = "number of batches")
    parser.add_argument("--epochs", type = int, default = 10, help = "number of epochs to train model")
    parser.add_argument("--beta", type = float, default = None, help = "To train a beta vae")
    parser.add_argument("--gen_gp_on_fly", action = "store_true", help = "at every epoch sample a new batch of gp's during training")
    parser.add_argument("--popw",action = "store_true", help = "normalise gp by population")
    args = parser.parse_args()

    # ----------------------------- Save Destination ----------------------------- #
    
    fxd_or_fly = "fly" if args.gen_gp_on_fly else "fxd"
    n_total = args.n_samples * args.n_batches
    popn = "popw" if args.popw else "nopopw"

    if args.beta:
        save_path = os.path.join(
            "model_runs/py/v4/c9s46s3077",
            f"aggvae_ep{args.epochs}_sb{int(n_total/1000)}k_{popn}_beta_{fxd_or_fly}"
        )
    else:
        save_path = os.path.join(
            "model_runs/py/v4/c9s46s3077",
            f"aggvae_ep{args.epochs}_sb{int(n_total/1000)}k_{popn}_{fxd_or_fly}"
        )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ----------------------------------- Data ----------------------------------- #

    gdf_census = gpd.read_file(os.path.join(args.data_root,"census9_state46_county3077","census9","census9.shp"))
    gdf_state = gpd.read_file(os.path.join(args.data_root,"census9_state46_county3077","state46", "state46.shp"))
    gdf_county = gpd.read_file(os.path.join(args.data_root,"census9_state46_county3077","county3077","county3077.shp"))

    coords = gdf_county.filter(items = ["centroid_x","centroid_y"])     
    pol_pts_lo, pt_which_pol_lo = compute_pts_in_polys(coords,gdf_census)
    pol_pts_hi, pt_which_pol_hi = compute_pts_in_polys(coords,gdf_state)

    # Lat/Lon points at county level
    x = jnp.array(coords) #(n_pts,2)
    n_pts,_ = x.shape
    pop = gdf_county.pop2020.values if args.popw else None

    samples_key, svi_key, split_key = random.split(PRNGKey(0), 3)

    # ---------------------------------- GP aggr --------------------------------- #
  
    gp_aggr_predictive = Predictive(
        gp_aggr,
        num_samples = args.n_samples if args.gen_gp_on_fly else args.n_samples * args.n_batches
    )
    gp_aggr_samples = gp_aggr_predictive(
        samples_key, 
        x = x, 
        M_lo = pol_pts_lo, 
        M_hi = pol_pts_hi, 
        popw = pop, 
        jitter = 1e-4, 
        noise = 1e-4
    )["gp_aggr"]

    # ------------------------------------ SVI ----------------------------------- #
    # optimizer 
    adam = numpyro.optim.Adam(step_size = 0.001)
    # stochastic variational inference 
    svi = SVI(
        model, 
        guide, 
        adam, 
        RenyiELBO(), 
        hidden_dim = 50, 
        z_dim = 40,
        beta = args.beta
    )
    svi_state = svi.init(svi_key, gp_aggr_samples)

    # ---------------------------------- Metric ---------------------------------- #
    losses = {
        "train" : [],
        "valid" : []
    }

    if args.gen_gp_on_fly:
        # generate keys 
        train_key, eval_key, samples_key = random.split(split_key, 3)
   
        # train validation sizes 
        train_size = int(0.7 * args.n_batches)
        valid_size = args.n_batches - train_size 

        # Train 
        for e in range(args.epochs): 
            train_loss, svi_state = epoch_train_gen_gp_on_fly(
                svi_state, 
                train_key, 
                num_samples = args.n_samples, 
                num_batches = args.n_batches,
                x = x, 
                M_lo = pol_pts_lo,
                M_hi = pol_pts_hi,
                popw = pop
            )
            valid_loss = epoch_eval_gen_gp_on_fly(
                svi_state, 
                eval_key,
                num_samples = args.n_samples, 
                num_batches = args.n_batches,
                x = x, 
                M_lo = pol_pts_lo, 
                M_hi = pol_pts_hi,
                popw = pop
            )
            losses["train"].append(train_loss.item())
            losses["valid"].append(valid_loss.item())
            print(f"epoch : {e}, training loss : {train_loss.item()}, valid loss : {valid_loss.item()}")
    else:

        # Torch tensordataset
        dataset = TensorDataset(
            torch.from_numpy(np.array(gp_aggr_samples))
        )
        # Train/Validation sizes
        train_size = int(len(dataset) * 0.7)
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        # Dataloaders
        trainloader = DataLoader(train_dataset, batch_size=64, collate_fn=jax_collate, drop_last = True)
        validloader = DataLoader(valid_dataset, batch_size=64, collate_fn=jax_collate, drop_last = True)

        for e in range(args.epochs):
            svi_state, train_loss = epoch_train_fixed_gps(svi_state, trainloader)
            valid_loss = epoch_eval_fixed_gps(svi_state, validloader)
            
            losses["train"].append(train_loss)
            losses["valid"].append(valid_loss)
            print(f"train loss: {train_loss:.2f}, valid loss : {valid_loss:.2f}")
    
    # Save decoder
    dec_params = svi.get_params(svi_state)
    
 
    with open(os.path.join(save_path, "dec_wts"),"wb") as file:
        pickle.dump(dec_params, file)

    # Losses
    p = figure(title = "losses",background_fill_color = "#fafafa")
    p.xaxis.axis_label = "epochs"
    p.yaxis.axis_label = "loss"
    p.line(np.arange(args.epochs),losses["train"], color = "coral", legend_label = "train")
    p.line(np.arange(args.epochs),losses["valid"], color = "gold", legend_label = "valid")
    #show(p)
    export_png(p, filename = os.path.join(save_path,"losses.png"))

    
