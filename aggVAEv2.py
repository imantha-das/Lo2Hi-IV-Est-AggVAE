# ---------------------------------------------------------------------------- #
#                 Approximate GP with VAE to speed up training                 #
# ---------------------------------------------------------------------------- #

import os
from pathlib import Path 
import numpy as np
import jax.numpy as jnp 
import jax.random as random
import jax
from jax import lax, jit 
from jax.example_libraries import stax 
from jaxtyping import Array,Float, Integer,Bool

import torch 
from torch.utils.data import TensorDataset, DataLoader

import numpyro  
from numpyro.infer import SVI, MCMC, NUTS, Predictive, RenyiELBO
import numpyro.distributions as dist 

import plotly.express as px 

from termcolor import colored 
from utils import get_comp_grid, exp_sq_kernel, M_g, plot_agg_gps
import argparse
import pickle

# ------------------------------- Aggragted GP ------------------------------- #

def gp_aggr(
    x:Float[Array, "n_grd_pts 2"],
    M_lo:Float[Array, "n_regions n_grid_pts"],
    M_hi:Float[Array, "n_regions n_grid_pts"],
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
        M_g(M_lo,f)
    ) # (n_regions_lo,)
    gp_aggr_hi = numpyro.deterministic(
        "gp_aggr_hi",
        M_g(M_hi,f),
    ) # (n_regions_hi,)
    gp_aggr = numpyro.deterministic(
        "gp_aggr",
        jnp.concatenate([gp_aggr_lo, gp_aggr_hi], axis = 0)
    ) # (n_regions_lo+ho,)
    return gp_aggr #(58,)

# ---------------------------------- Encoder --------------------------------- #
def encoder(hidden_dim = 50, z_dim = 40):
    return stax.serial(
        #(n_samples, n_regions) -> (n_samples, hidden_dim)
        stax.Dense(hidden_dim, W_init = stax.randn()), #(5,58) -> (5,50)
        stax.Elu, # try different one such as stax.Softplus
        stax.FanOut(2),
        stax.parallel(
            # mean : (n_samples, hidden_dim) -> (n_samples, z_dim)
            stax.Dense(z_dim, W_init = stax.randn()), #(5,50) -> (5, 40)
            # std : (n_samples, hidden_dim) -> (n_samples, z_dim)
            stax.serial(
                stax.Dense(z_dim, W_init = stax.randn()),
                stax.Exp
            )
        )
    )

def guide(batch, hidden_dim = 50, z_dim = 40):
    """This is the encoder portion of the VAE""" 
    assert batch.ndim == 2, "batch must have shape (n_samples, n_regions)"
    n_samples, input_dim = batch.shape # n_samples, n_regions 
    encode = numpyro.module(
        name = "encoder",
        nn = encoder(hidden_dim = hidden_dim, z_dim = z_dim),
        input_shape = (n_samples, input_dim)
    )
    z_loc, z_std = encode(batch)

    with numpyro.plate("batch",n_samples): #note n_samples is batch dimension
        return numpyro.sample(
            "z", 
            dist.Normal(z_loc,z_std).to_event(1)
        )

# ---------------------------------- Decoder --------------------------------- #
def decoder(hidden_dim, out_dim):
    return stax.serial(
        #(n_samples, z_dim) -> (n_samples, hidden_dim)
        stax.Dense(hidden_dim, W_init = stax.randn()), #(5,40) -> (5,50)
        stax.Elu,
        #(n_samples, hidden_dim) -> (n_samples, n_regions)
        stax.Dense(out_dim, W_init = stax.randn()) #(5, 50) -> (5,58)
    )

def model(batch, hidden_dim = 50, z_dim = 40):
    """This is the decoder part of the VAE""" 
    assert batch.ndim == 2, "batch must have a shape (n_samples, n_regions)"
    n_samples, out_dim = batch.shape # 5, 58

    decode = numpyro.module(
        name = "decoder",
        nn = decoder(hidden_dim = hidden_dim, out_dim = out_dim),
        input_shape = (n_samples, z_dim)
    )
    with numpyro.plate("batch", n_samples):
        z = numpyro.sample(
            "z", 
            dist.Normal(
                jnp.zeros((n_samples, z_dim)),
                jnp.ones((n_samples, z_dim))
            ).to_event(1)
        )

        region_mu = decode(z) #(n_regions,)
        
        return numpyro.sample(
            "obs",
            dist.Normal(region_mu, 1).to_event(1), #hardcode variance as 1
            #* We need to show the batches during training 
            #* During inference we might set this to None - check
            obs = batch 
        )

# ------------------------------- Training Loop ------------------------------ #


def epoch_train_fixed_gps(svi_state, trainloader):
    loss_sum = 0.0
    for i, (gp_batch,) in enumerate(trainloader):  # note: (gp_batch,) because of TensorDataset
        #rng_key_i = jax.random.fold_in(rng_key, i)
        svi_state, loss = svi.update(svi_state, gp_batch)
        num_samples = gp_batch.shape[0]
        loss_sum += loss / num_samples
    return svi_state, loss_sum / len(trainloader)

def epoch_eval_fixed_gps(svi_state, valloader):
    loss_sum = 0.0 
    for i, (gp_batch,) in enumerate(valloader):
        num_samples = gp_batch.shape[0]
        # svi_state is returned by evaluate but we dont need it
        loss = svi.evaluate(svi_state, gp_batch)
        loss_sum += loss / num_samples
    return loss_sum / len(valloader)

@jax.jit 
def epoch_train_gen_gp_on_fly(
    svi_state, 
    rng_key, 
    num_samples, 
    num_batches,
    x, 
    M_lo, M_hi
    ):
    """
    Rather than computing a large set of GP's and storing them in a dataloader.
    We generate them on fly inside the funtion. 
    During each epoch a different set of samples will be seen by the model.
    This approach should be noisy. 
    """
    def body_fn(i, val):
        # get total loss and svi state
        loss_sum, svi_state = val 
        # we use random.fold_in a method to generate keys 
        rng_key_i = jax.random.fold_in(rng_key, i)
        # Everytime we instantiate new set of GP samples 
        #gp_predictive = Predictive(gp_aggr, num_samples = num_samples)
        gp_samples = gp_predictive(
            rng_key_i,
            x = x, 
            M_lo = M_lo,
            M_hi = M_hi
        )["gp_aggr"] #(n_samples, n_regions)
        # Get svi state and loss
        svi_state, loss = svi.update(svi_state, gp_samples)
        # accumulate per sample losses
        loss_sum += loss / num_samples # This will be done num_batches times 
        return loss_sum, svi_state

    loss, svi_state = lax.fori_loop(
        lower = 0, 
        upper = num_batches, 
        body_fun = body_fn,
        init_val= (0.0, svi_state)
    )
    return loss / num_batches , svi_state

@jit 
def epoch_eval_gen_gp_on_fly(
    svi_state,
    rng_key,
    num_samples,
    num_batches,
    x,
    M_lo, M_hi
    ):
    """
    Rather than computing a large set of GP's and storing them in a dataloader.
    We generate the samples on the fly every single epoch. 
    """
    def body_fn(i, loss_sum):
        # Use random_fold_in method generate keys without any side effects 
        rng_key_i = jax.random.fold_in(rng_key, i)
        # Generate GP samples
        gp_samples = gp_predictive(
            rng_key_i,
            x = x, 
            M_lo = M_lo,
            M_hi = M_hi
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

# ----------------------- Pytorch Tensors to Jax Arrays ---------------------- #
def jax_collate(batch):
    """Converts a torch tensor to a Jax array"""
    return jax.tree_util.tree_map(
        jnp.array, 
        torch.utils.data.default_collate(batch)
    )

if __name__ == "__main__":
    # ------------------------------ Argument Parser ----------------------------- #
    
    parser = argparse.ArgumentParser(description = "Run VaeGP model")
    parser.add_argument("--data_path", type = str, default = "data/processed", help = "path to where data is stored")
    parser.add_argument("--hidden_dim", type = int, default = 50, help = "hidden dimension of encoder and decoder")
    parser.add_argument("--z_dim", type = int, default = 40, help = "latent dimension size")
    parser.add_argument("--n_samples", type = int, default = 32, help = "number of gp samples")
    parser.add_argument("--n_batches", type = int, default = 10, help = "number of batches")
    parser.add_argument("--epochs", type = int, default = 10, help = "number epochs")
    parser.add_argument(
        "--gen_gp_on_fly", action = "store_true", 
        help = "generate gps inside train function instead of passing them as a trainloader"
    )
    args = parser.parse_args()

    # -------------------------- Data and Saved Vectors -------------------------- #
    
    grid_data = get_comp_grid(Path(args.data_path))
    # Load necessay saved vectors
    x:Float[Array, "n_grid_pts 2"] = jnp.array(grid_data["x"])
    pol_pt_lo:Float[Array, "n_regions n_grid_pts"] = jnp.array(grid_data["pol_pt_lo"])
    pt_which_pol_lo:Float[Array, "n_grid_pts"] = jnp.array(grid_data["pt_which_pol_lo"])
    pol_pt_hi:Float[Array, "n_regions n_grid_pts"] = jnp.array(grid_data["pol_pt_hi"])
    pt_which_pol_hi:Float[Array, "n_grid_pts"] = jnp.array(grid_data["pt_which_pol_hi"])
    
    # ------------------------ Prior Predictive Simulation ----------------------- #

    rng_key_prior, rng_subkey = random.split(random.PRNGKey(3))
    gp_prior_predictive = Predictive(gp_aggr, num_samples = 100)
    agg_gp_prior = gp_prior_predictive(
        rng_key_prior, 
        x = x,
        M_lo = pol_pt_lo,
        M_hi = pol_pt_hi
    )["gp_aggr"] #(num_samples,58)

    # n_samples, f_dims = agg_gp_draws.shape #5 , 58
    # #plot_agg_gps(prior_gp_draws)

    # ------------------------------- Initiate SVI ------------------------------- #

    adam = numpyro.optim.Adam(step_size = 0.001)

    svi = SVI(
        model,
        guide,
        adam,
        RenyiELBO(),
        hidden_dim = args.hidden_dim,
        z_dim = args.z_dim
    )
    
    svi_key, subkey = random.split(random.PRNGKey(123),2)
    svi_state = svi.init(svi_key, agg_gp_prior)
    losses = {"train" : [], "valid" : []}
    if args.gen_gp_on_fly:
        # ------------------ Train model by sampling Gps on the fly ------------------ #
        # We will be generating samples for inside the function - for batch appraoch look at the else statement
        # We cant instantiate the predictive function inside the train or validate functions, so we will instantiate it outside
        gp_predictive = Predictive(
            gp_aggr, 
            num_samples = args.n_samples 
        )
        train_key, eval_key, rng_subkey = random.split(subkey,3)
        # note we since we are generating n_samples inside the function, we will be this n_batches times
        n_train= int(0.7 * args.n_batches) # we will be training n_train in a single epoch, this equal to the number of batches
        n_valid = args.n_batches - n_train 
        # Train
        for e in range(args.epochs):
            train_loss, svi_state = epoch_train_gen_gp_on_fly(
                svi_state,
                train_key,
                num_samples = args.n_samples,
                num_batches = n_train,
                x = x, 
                M_lo = pol_pt_lo,
                M_hi = pol_pt_hi
            )
            valid_loss = epoch_eval_gen_gp_on_fly(
                svi_state, 
                eval_key,
                num_samples = args.n_samples,
                num_batches = n_valid, 
                x = x,
                M_lo = pol_pt_lo,
                M_hi = pol_pt_hi
            )
            losses["train"].append("train_loss") 
            losses["valid"].append("valid_loss")
            print(f"train loss : {train_loss:.3f}, valid loss : {valid_loss:.3f}")
    else:
        # --------------------- Train Model on fixed set of GP's --------------------- #
        # Generate GP samples which and store them in a dataloader. 
        # FOr every epoch the model will see the same set of GP's 
        gp_predictive = Predictive(
            gp_aggr, 
            num_samples = args.n_samples * args.n_batches
        )
        predictive_key, rng_subkey = random.split(rng_subkey)
        agg_gp_draws = gp_predictive(
            predictive_key, 
            x = x, 
            M_lo = pol_pt_lo, 
            M_hi = pol_pt_hi
        )["gp_aggr"]
        # Convert them to numpy as they need to be converted to torch.tensors
        agg_gp_draws = np.array(agg_gp_draws)
        agg_gp_draws_tch = torch.from_numpy(agg_gp_draws) 
        # Torch dataset and dataloader
        dataset = TensorDataset(agg_gp_draws_tch)
        train_size = int(len(dataset) * 0.7)
        valid_size = len(dataset) - train_size 
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
        trainloader = DataLoader(train_dataset, batch_size = args.n_samples, collate_fn = jax_collate, drop_last = True)
        validloader = DataLoader(valid_dataset, batch_size = args.n_samples, collate_fn = jax_collate, drop_last = True)

        for i in range(args.epochs):
            svi_state, train_loss = epoch_train_fixed_gps(svi_state,trainloader)
            valid_loss = epoch_eval_fixed_gps(svi_state,validloader)
            losses["train"].append("train_loss") 
            losses["valid"].append("valid_loss")
            print(f"train loss : {train_loss:.2f}, valid loss : {valid_loss:.2f}")

        # ---------------------------- Save Model Weights ---------------------------- #
    # save decoder 
    if not os.path.exists("model_weights"):
        os.mkdir("model_weights")
    decoder_params = svi.get_params(svi_state)
    fixed_or_fly = "fly" if args.gen_gp_on_fly else "fixed"
    f_name = f"aggvae_dec_ep{args.epochs}_h{args.hidden_dim}_z{args.z_dim}_{fixed_or_fly}"
    print("Saving decoder params in 'model_weighs'...")
    with open(os.path.join("model_weights",f_name), "wb") as file:
        pickle.dump(decoder_params, file)