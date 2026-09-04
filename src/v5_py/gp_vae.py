# ==============================================================================
# Learn to Generate Gaussian Process Spatial Priors through Variational Autoencoders
# ==============================================================================
import os

import jax
import jax.numpy as jnp 
import jax.random as random
from jaxtyping import Float, Array
from typing import Union
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import flax_module
from numpyro.infer import Predictive, SVI, RenyiELBO

import flax.linen as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt

import pickle
import yaml

from utils import exp_sq_kernel, compute_pts_in_polys


# ==============================================================================
# Functions
# ==============================================================================
# ------------------------------------------------------------------------------
# Gaussian Process
# ------------------------------------------------------------------------------

def gp(x:Float[Array, "n_pts 2"]):
    # kernel length
    kl = numpyro.sample("kl", dist.InverseGamma(3,3))
    # kernel variance 
    kv = numpyro.sample("kv", dist.HalfNormal(0.5))
    # GP kernel
    k = exp_sq_kernel(x,x,kv,kl,noise = 1e-4,jitter = 1e-4)
    # Sample a GP
    f = numpyro.sample(
        "f",
        dist.MultivariateNormal(
            loc = jnp.zeros(x.shape[0]), #(n_grid_pts,)
            covariance_matrix = k #(n_grid_pts, n_grid_pts)
        ),
        obs = None
    ) # (n_grid_pts,)
    return f

# ------------------------------------------------------------------------------
# Variational Autoencoder
# ------------------------------------------------------------------------------

# class Encoder(nn.Module):
#     """VAE Encoder""" 
#     input_dims: int = 3077 # Input dimensions, no of counties 

#     @nn.compact
#     def __call__(self, x):
#         """ x : (*, 3077)"""
#         h1_dims = self.input_dims // 2 # int : 1538
#         h2_dims = h1_dims // 2 #int : 769
#         z_dims = h2_dims // 2 # int : 384
#         jitter = 1e-5
#         # Hidden dimension 
#         h1_out = nn.elu(nn.Dense(h1_dims)(x)) # (*, 1538)
#         h2_out = nn.elu(nn.Dense(h2_dims)(h1_out)) # (*, 769)
#         # mu and sigma of gp 
#         z_loc = nn.Dense(z_dims)(h2_out)
#         z_scale = nn.softplus(nn.Dense(z_dims)(h2_out)) + jitter
#         return z_loc, z_scale

# class Decoder(nn.Module):
#     """VAE Decoder"""
#     input_dims: int = 3077

#     @nn.compact
#     def __call__(self, z):
#         """z shape : (*,384)"""
#         h1_dims = self.input_dims // 2 # int : 1538
#         h2_dims = h1_dims // 2 # int : 769
#         # Reconstruct GP's
#         h2_out = nn.elu(nn.Dense(h2_dims)(z)) # (*, 769)
#         h1_out = nn.elu(nn.Dense(h1_dims)(h2_out)) #(*, 1538)
#         x_out = nn.Dense(self.input_dims)(h1_out) #(*, 3077)
#         return x_out

# ------------------------------------------------------------------------------
class Encoder(nn.Module):
    """VAE Encoder""" 
    params: dict  # model parameters (hidden layers, latent dimension etc.)

    @nn.compact
    def __call__(self, x):
        """ x : (*, 3077)"""
        h1_dims = self.params["h1_dims"] # int : 1538
        h2_dims = self.params["h2_dims"] #int : 769
        z_dims = self.params["z_dims"] # int : 384
        jitter = self.params["jitter"]
        # Hidden dimension 
        h1_out = nn.elu(nn.Dense(h1_dims)(x)) # (*, 1538)
        h2_out = nn.elu(nn.Dense(h2_dims)(h1_out)) # (*, 769)
        # mu and sigma of gp 
        z_loc = nn.Dense(z_dims)(h2_out)
        z_scale = nn.softplus(nn.Dense(z_dims)(h2_out)) + jitter
        return z_loc, z_scale

class Decoder(nn.Module):
    """VAE Decoder"""
    params: dict # model parameters (hidden layers, latent dimensions etc.)

    @nn.compact
    def __call__(self, z):
        """z shape : (*,384)"""
        input_dims = self.params["input_dims"] # int : 3077
        h1_dims = self.params["h1_dims"] # int : 1538
        h2_dims = self.params["h2_dims"] # int : 769
        # Reconstruct GP's
        h2_out = nn.elu(nn.Dense(h2_dims)(z)) # (*, 769)
        h1_out = nn.elu(nn.Dense(h1_dims)(h2_out)) #(*, 1538)
        x_out = nn.Dense(input_dims)(h1_out) #(*, 3077)
        return x_out

# ------------------------------------------------------------------------------
# Guide + Model
# ------------------------------------------------------------------------------

def guide(batch, params):
    """This is the encoder part of the VAE""" 
    n_samples, input_dims = batch.shape # i.e 100 , 3077 
    encode = numpyro.contrib.module.flax_module(
        name = "encoder",
        nn_module = Encoder(params),
        input_shape = (n_samples, input_dims)
    )
    z_loc, z_std = encode(batch)
    # Beta VAE 
    if params["beta"]: 
        with numpyro.plate("batch", n_samples):
            with numpyro.handlers.scale(scale = params["beta"]):
                return numpyro.sample(
                    "z",
                    dist.Normal(z_loc,z_std).to_event(1)
                )
    # Regular VAE 
    else: 
        # note n_samples is the batch size 
        # usage of "plate" is to ensure the latent vector z are independent.
        with numpyro.plate("batch", n_samples):
            # to event reats z_dim (384) indpendent distributions as one vector containing 384 distributions 
            return numpyro.sample(
                "z",
                dist.Normal(z_loc, z_std).to_event(1)
            )

def model(batch, params):
    """ This is the decoder part of the VAE """ 
    # This is still the same GP batch not z (event though the decoder uses z)
    n_samples, input_dims = batch.shape
    z_dim = params["z_dims"]
    decode = numpyro.contrib.module.flax_module(
        name = "decoder",
        nn_module = Decoder(params),
        input_shape=(1,z_dim)
    )
    if params["beta"]:
        with numpyro.plate("batch", n_samples):
            with numpyro.handlers.scale(scale = params["beta"]):
                z = numpyro.sample(
                    "z",
                    dist.Normal(
                        jnp.zeros((n_samples, z_dim)),
                        jnp.ones((n_samples, z_dim))
                    ).to_event(1)
                )
                regions_mu = decode(z) # (*,3077)
                return numpyro.sample(
                    "obs",
                    dist.Normal(regions_mu, 1).to_event(1),
                    obs = batch 
                )
    else:
        with numpyro.plate("batch", n_samples):
            z = numpyro.sample(
                "z",
                dist.Normal(
                    jnp.zeros((n_samples, z_dim)),
                    jnp.ones((n_samples, z_dim))
                ).to_event(1)
            )
            regions_mu = decode(z) # (*,3077)
            return numpyro.sample(
                "obs",
                dist.Normal(regions_mu, 1).to_event(1),
                obs = batch
            )
        
@jax.jit 
def epoch_train_gen_gp_on_fly(
    svi_state, rng_key, num_batches, x, global_scale
):
    """ Generate GP's on fly - at every epoch the model sees a new set of GP's """
    def body_fn(i, val):
        # get total loss and svi-state
        loss_sum, svi_state = val 
        # we use random.fold_in method to generate keys 
        rng_key_i = jax.random.fold_in(rng_key, i)
        # Instantiate a new set of gp's 
        gp_samples = gp_predictive(rng_key_i,x = x)["f"] #(*, n_pts)
        gp_samples_norm = gp_samples / global_scale
        num_samples = gp_samples.shape[0]
        # SVI state and loss 
        svi_state, loss = svi.update(svi_state, gp_samples_norm)
        # Accumulate per loss sample 
        loss_sum += loss / num_samples 
        return loss_sum, svi_state 

    loss,svi_state = jax.lax.fori_loop(
        lower = 0,
        upper = num_batches,
        body_fun = body_fn, 
        init_val = (0.0, svi_state)
    )
    return loss/num_batches , svi_state

# ------------------------------------------------------------------------------
# Plot spatial Fields
# ------------------------------------------------------------------------------

def plot_spf_priors(gdf,dec_params,split_key,global_scale, model_params, train_params, save_path):
    samples_key, z_key, split_key = random.split(split_key,3)
    n_samples = train_params["n_samples"]
    z_dims = model_params["z_dims"]
    # Create a bunch of GP's
    gp_predictive = Predictive(gp, num_samples = n_samples)
    gp_samples = gp_predictive(samples_key, x = x)["f"] #(n_samples, n_pts)
    gdf["gp_mean"] = gp_samples.mean(axis = 0) #(1, n_pts)
    # VAE reconstructed GP's
    z = dist.Normal(jnp.zeros(z_dims), jnp.ones(z_dims)).sample(z_key, sample_shape = (n_samples,)) #(z_dim,)
    decoder = Decoder(params = model_params)
    gp_recon_norm = decoder.apply({"params" : dec_params}, z) #(n_samples,n_pts) , samples produce by VAE are normalize ones
    gdf["gp_recon_mean"] = gp_recon_norm.mean(axis = 0) * global_scale

    # Plot GPs
    fig,ax = plt.subplots(1,2, figsize = (20,6))
    gdf.plot(
        column = "gp_mean",
        cmap = "coolwarm",
        legend = True,
        ax = ax[0],
        edgecolor="black"
    )
    ax[0].set_title("True GP field Prior")
    gdf.plot(
        column = "gp_recon_mean",
        cmap = "coolwarm",
        legend = True,
        ax = ax[1],
        edgecolor = "black"
    )
    ax[1].set_title("VAE reconstructe GP field Prior")
    fig.savefig(
        os.path.join(save_path, "gp_vs_vae_recon_spf.png")
    )

# ==============================================================================
# Main
# ==============================================================================
data_root = "data/processed/v4_clin_labs"
gdf_county = gpd.read_file(os.path.join(data_root, "census9_state46_county3077","county3077","county3077.shp"))
with open("src/v5_py/config.yml") as f:
    params = yaml.safe_load(f)

model_params = params["model"]
train_params = params["training"]

coords = gdf_county.filter(items = ["centroid_x","centroid_y"]) #(3077,2)
x = jnp.array(coords)
n_pts, _ = x.shape
n_samples = train_params["n_samples"]


svi_key, train_key, eval_key, samples_key, split_key = random.split(random.PRNGKey(0), 5)

gp_predictive = Predictive(gp, num_samples = n_samples)
gp_samples = gp_predictive(samples_key, x = x)["f"] #(*, n_pts)
global_scale = gp_samples.std()

#? ------------------------------------------------------------------------------
#? Testing code ...
#? ------------------------------------------------------------------------------

# # Plot this in a map instead to check 
# global_scale = gp_samples.std()
# gp_samples_norm = gp_samples / global_scale
# gp_mean = gp_samples.mean(axis = 0)
# gp_mean_norm = gp_samples_norm.mean(axis = 0)
# gdf_county["gp_mean"] = gp_mean
# gdf_county["gp_mean_norm"] = gp_mean_norm
# fig,ax = plt.subplots(1,1, figsize = (20,6))
# gdf_county.plot(
#     column = "gp_mean_norm",
#     cmap = "coolwarm",
#     legend = True,
#     ax = ax,
#     edgecolor="black"
# )

# # Test encoder
# encoder = Encoder(params = model_params)
# enc_params = encoder.init(random.PRNGKey(0), gp_samples[0,:])
# z_loc,z_scale = encoder.apply(enc_params, gp_samples)

# # Test decoder
# # Note we cant use MvN since we need a covaraince of shape (n_pts, n_pts) but our z_scale is (*, n_pts)
# z = dist.Normal(z_loc, z_scale).to_event(1).sample(random.PRNGKey(0))
# decoder = Decoder(params = model_params)
# dec_params = decoder.init(random.PRNGKey(0), z[0, :])
# gp_out = decoder.apply(dec_params, z)

# # Test Guide 
# guide_predictive = Predictive(guide, num_samples=(1))
# guide_out = guide_predictive(
#     random.PRNGKey(0), 
#     batch = gp_samples, 
#     params = model_params
# )["z"].squeeze(0)# squeeze to (1,100,384) -> (100,384)

# model_predictive = Predictive(model, num_samples=(1)) 
# model_out = model_predictive(
#     random.PRNGKey(0),
#     batch = gp_samples,
#     params = model_params
# )["obs"].squeeze(0) # squeeze to (1,100,3077) -> (100,3077)

#? ----------------------------------------------------------------------------- 
# ------------------------------------------------------------------------------
# VAE training loop
# ------------------------------------------------------------------------------

print(jax.default_backend())


print(global_scale)
# optimizer
optimizer = numpyro.optim.Adam(step_size = train_params["lr"])
# Instantiate Stochastic Variational Inference 
svi = SVI(
    model, guide, optimizer, RenyiELBO(), params = model_params
)
svi_state = svi.init(svi_key, gp_samples)
losses = {"train" : [], "valid" : []}
for e in range(20):
    train_loss, svi_state = epoch_train_gen_gp_on_fly(
        svi_state = svi_state, 
        rng_key = train_key,
        num_batches = 100,
        x = x,
        global_scale = global_scale
    )
    losses["train"].append(train_loss)
    print(f"epoch : {e}, loss : {train_loss}")

# Create Save folder
save_fold = f"gp_vae_2lyr_h{model_params["h1_dims"]}_h{model_params["h2_dims"]}_z{model_params["z_dims"]}_ep{train_params["epochs"]}_bta{model_params["beta"]}"
save_path = os.path.join("model_outputs/v5",save_fold)
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Save losses
pd.DataFrame({
    "epochs" : jnp.arange(train_params["epochs"]),
    "train_loss" : losses["train"]
}).to_csv(os.path.join(save_path, "loss.csv"), index = False)

# Save decoder weights
dec_params = svi.get_params(svi_state)["decoder$params"]
with open(os.path.join(save_path, "dec_wts"),"wb") as f: 
    pickle.dump(dec_params, f)

# ------------------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------------------

with open(os.path.join(save_path, "dec_wts"),"rb") as f:
    dec_params = pickle.load(f)

plot_spf_priors(gdf_county,dec_params, split_key,global_scale, model_params,train_params, save_path)



