# ==============================================================================
# Script to run Aggregated Variational AutoEncoder (aggVAE) on Influenza Cases 
# at differnt resolution administrative boundaries.
# ==============================================================================
# ---------------------------------- imports --------------------------------- #

import os
import math
import numpy as np 

import time

import jax 
import jax.numpy as jnp 
from jax import random, lax, jit, ops
from jax.example_libraries import stax

import numpyro 
from numpyro.infer import SVI, MCMC, NUTS, init_to_median, Predictive, RenyiELBO 
import numpyro.distributions as dist  

import geopandas as gpd
import plotly.express as px

from termcolor import colored
from aggGP import dist_euclid, exp_sq_kernel, M_g

import pickle

jax.config.update("jax_default_device", jax.devices()[1])
print(f"Jax using device : {jax.devices()}")


# ------------------- Func for Prior Predictive Simulation ------------------- #
def gp_aggr(args):
    x = args["x"] # (num_grid_pts, lat+lon) <- (2618,2)
    gp_kernel = args["gp_kernel"]
    noise = args["noise"]
    
    M_lo= args["M_lo"] # (9, 2618)
    M_hi = args["M_hi"] # (49, 2618), 

    #kernal hyperparams
    kernal_length = args["kernel_length"]
    kernel_var = args["kernel_var"]

    # Random effect - aggregated GP 
    length = numpyro.sample("kernel_length", kernal_length) #(,)
    var = numpyro.sample("kernel_var",kernel_var) #(,)
    # Kernel for allgrid points
    k = gp_kernel(x,x,var, length, noise) #(num_grig_pts,num_grid_pts) <- (2618,2618)
    # GP draw evaluated at all 2618 grid pints
    f = numpyro.sample(
        "f", 
        dist.MultivariateNormal(loc = jnp.zeros(x.shape[0]), covariance_matrix = k)
        ) #(num_grid_pts,) <- i.e (2618,)

    #aggregate f into gp_aggr according to indexing of (point in polygon)
    gp_aggr_lo = numpyro.deterministic("gp_aggr_lo", M_g(M_lo, f)) #(num_regions,) <- i.e (9,) for lo
    gp_aggr_hi = numpyro.deterministic("gp_aggr_hi", M_g(M_hi, f)) #(49,)
    gp_aggr = numpyro.deterministic("gp_aggr", jnp.concatenate([gp_aggr_lo, gp_aggr_hi])) #(58,)
    return gp_aggr

    
# -------------------------- Variational Autoencoder ------------------------- #
def vae_encoder(hidden_dim = 50, z_dim = 40):
    return stax.serial(
        #(num_samples, num_regions) -> (num_samples, hidden_dims) 
        stax.Dense(hidden_dim, W_init = stax.randn()), #i.e(5,58) -> (5,50)
        stax.Elu,
        stax.FanOut(2),
        stax.parallel(
            # mean : (num_samples, hidden_dim) -> (num_samples, z_dim)
            stax.Dense(z_dim, W_init = stax.randn()), #(5,50) -> (5,40)
            #std : (num_samples, hidden_dim) -> (num_samples, z_dim)
            stax.serial(stax.Dense(z_dim, W_init = stax.randn()), stax.Exp) #(5,50) -> (5,40)
        )
    )

def vae_decoder(hidden_dim, out_dim):
    return stax.serial(
        # (num_samples, z_dim) -> (num_samples, hidden_dim): (5,40) -> (5,50)
        stax.Dense(hidden_dim, W_init = stax.randn()),
        stax.Elu,
        # (num_samples, hidden_dim) -> (num_samples, num_regions) : (5,50) -> (5, 58)
        stax.Dense(out_dim, W_init = stax.randn())
    )


def vae_model(batch, hidden_dim, z_dim):
    """This computes the decoder portion"""
    batch = jnp.reshape(batch, (batch.shape[0], -1)) # (num_samples, num_regions) <- i.e (5,58) 
    batch_dim, out_dim = jnp.shape(batch) # 5 , 58 

    # vae-decoder in numpyro module
    decode = numpyro.module(
        name = "decoder", 
        nn = vae_decoder(hidden_dim = hidden_dim, out_dim = out_dim),
        input_shape = (batch_dim, z_dim) #(5,40) 
    )

    # Sample a univariate normal
    #! ISSUE HERE : lax.sub cannot broadcast shapes (5,40) & (40,) here 
    #! SO HAD TO CHANGE dist.Normal(jnp.zeros((z_dim,)), jnp.ones((z_dim,))) TO THIS dist.Normal(jnp.zeros((5,z_dim)), jnp.ones((5,z_dim)))
    z = numpyro.sample(
        "z", 
        dist.Normal(
            jnp.zeros((batch_dim,z_dim)), 
            jnp.ones((batch_dim,z_dim))
            )
    ) # (z_dim,) : i.e (40,)
    # Forward pass from decoder
    gen_loc = decode(z) #(num_regions,) : (58,)
    obs = numpyro.sample(
        "obs", 
        dist.Normal(gen_loc, args["vae_var"]), 
        obs = batch
    ) #(num_samples, num_regions) : (5,58)
    return obs


def vae_guide(batch, hidden_dim, z_dim):
    """This computes the encoder portion"""
    batch = jnp.reshape(batch, (batch.shape[0], -1)) #(num_samples, num_regions) : (5,58)
    batch_dim, input_dim = jnp.shape(batch)# num_samples , num_regions : 5 , 58 

    # vae-encoder in numpyro module
    encode = numpyro.module(
        name = "encoder", 
        nn = vae_encoder(hidden_dim=hidden_dim,z_dim = z_dim),
        input_shape = (batch_dim, input_dim) #(5,58)
    ) #(num_samples, num_regions) -> (num_samples, hidden_dims) : i.e (5,58) -> (5,40)

    # Samapling mu, sigma - Pretty much the forward pass
    z_loc, z_std = encode(batch) #mu : (num_samples, z_dim), sigma2 : (num_samples, z_dim) <- (5,40),(5,40)
    # Sample a value z based on mu and sigma
    z = numpyro.sample("z", dist.Normal(z_loc, z_std)) #(num_sample, z_dim) : (5,40)
    return z



#! Something wrong with these two training functions
# @jit 
# def epoch_train(rng_key, svi_state, num_train):
#     def body_fn(i, val):
#         # Random keys
#         rng_key_i = random.fold_in(rng_key, i)
#         rng_key_i, rng_key_ls, rng_key_var, rng_key_noise = random.split(rng_key_i, 4)
#         loss_sum, svi_state = val 
#         # Gp Draw : (num_samples, num_regions)
#         batch = agg_gp_predictive(rng_key, args)["gp_aggr"] # (5,9)
#         # Forward pass, Takes "state" & "args" where args are your "model inputs" 
#         svi_state, loss = svi.update(svi_state, batch)
#         loss_sum += loss / args["batch_size"]
#         return (loss_sum / num_train), svi_state 
    
#     return lax.fori_loop(lower =0, upper = num_train, body_fun = body_fn, init_val = (0.0, svi_state))

# @jit
# def eval_test(rng_key, svi_state, num_test):
#     def body_fn(i, loss_sum):
#         rng_key_i = random.fold_in(rng_key, i)
#         rng_key_i, rng_key_ls, rng_key_var, rng_key_noise = random.split(rng_key_i, 4)
#         # GP Draw : (num_samples, num_regions) 
#         batch = agg_gp_predictive(rng_key_i, args)["gp_aggr"] # (5,49)
#         loss = svi.evaluate(svi_state, batch)
#         loss_sum += loss 
#         return loss 
#     loss = lax.fori_loop(0, num_test, body_fn, 0.0)
#     loss = loss / num_test

#     return loss

@jax.jit
def epoch_train(rng_key, svi_state, num_train):
    def body_fn(i, val):
        rng_key_i = jax.random.fold_in(rng_key, i) #Array(2,)
        rng_key_i, rng_key_ls, rng_key_var, rng_key_noise = jax.random.split(rng_key_i, 4) #Tuple(Array(2,) x 4)
        loss_sum, svi_state = val #val --svi_state
        
        batch = agg_gp_predictive(rng_key_i, args)["gp_aggr"] #(5,116) <- num_samples : 5, total_districts : 116
        #* svi is where the vae_model & vae_guide gets applied
        svi_state, loss = svi.update(svi_state, batch)
        loss_sum += loss / args["batch_size"]
        return loss_sum, svi_state 
    
    return lax.fori_loop(lower = 0, upper = num_train, body_fun=body_fn, init_val=(0.0, svi_state))

@jax.jit 
def eval_test(rng_key, svi_state, num_test):
    def body_fn(i, loss_sum):
        rng_key_i = jax.random.fold_in(rng_key, i)
        rng_key_i, rng_key_ls, rng_key_varm, rng_key_noise = jax.random.split(rng_key_i, 4)
        batch = agg_gp_predictive(rng_key_i, args)["gp_aggr"]
        #* svi is where the vae_model & vae_guide gets applied
        loss = svi.evaluate(svi_state, batch) / args["batch_size"]
        loss_sum += loss
        return loss_sum 
    
    loss = lax.fori_loop(lower = 0, upper = num_test,body_fun =  body_fn, init_val = 0.0)
    loss = loss / num_test 
    return loss

      
# -------------------- Function to plot a gaussian process ------------------- #
def plot_process(gp_draws):
    p = px.line()
    for i in range(len(gp_draws)):
        p.add_scatter(x = np.arange(gp_draws.shape[1]), y = gp_draws[i, :])

    p.update_traces(line_color = "black")
    p.update_layout(
        template = "plotly_white", 
        xaxis_title = "region", yaxis_title = "num cases",
        showlegend = False
    )
    p.show()


if __name__ == "__main__":
    # ------------------------------ Load variables ------------------------------ #
    # root directory
    var_dir = "data/processed"
    # Lat/Lon Values of artificial grid
    x = np.load(os.path.join(var_dir,"lat_lon_x.npy")) #(2618,2)
    # Low regional data
    pol_pt_lo = np.load(os.path.join(var_dir,"low","pol_pt_lo.npy")) #(9,2618)
    pt_which_pol_lo = np.load(os.path.join(var_dir,"low","pt_which_pol_lo.npy")) #(2618,)
    # High regional data 
    pol_pt_hi = np.load(os.path.join(var_dir,"high","pol_pt_hi.npy")) #(49, 2618,)
    pt_which_pol_hi = np.load(os.path.join(var_dir,"high","pt_which_pol_hi.npy")) #(2618,)
    # Dataframes
    df_lo = gpd.read_file(os.path.join(var_dir, "low", "us_census_divisions","us_census_divisions.shp"))
    df_hi = gpd.read_file(os.path.join(var_dir, "high", "us_state_divisions","us_state_divisions.shp"))


    #* ==========================================================================
    #  Variables that need changing

    #* ==========================================================================

    # --------------------------------- Arguments -------------------------------- #
    args = {
        "x": x,
        "gp_kernel": exp_sq_kernel,
        "noise": 1e-4,
        "M_lo": pol_pt_lo,
        "M_hi" : pol_pt_hi,

        # VAE training
        "rng_key": random.PRNGKey(5),
        "num_epochs": 20, 
        #"learning_rate": 1.0e-3, 
        "learning_rate": 0.0005, 
        "batch_size": 100, 
        "hidden_dim": 50, #! Hyperparam
        "z_dim": 40, #! Hyperparam
        "num_train": 100,
        "num_test":100,
        "vae_var": 1,
        # kernel hyperparams
        "kernel_length" : dist.InverseGamma(3,3), #!hyperparam
        "kernel_var" : dist.HalfNormal(0.05) #!hyperparam

    }

    # ------------------------ Prior Predictive Simulation ----------------------- #
    rng_key, rng_key_ = random.split(random.PRNGKey(4))
    agg_gp_predictive = Predictive(gp_aggr,num_samples = 5)

    agg_gp_draws = agg_gp_predictive(rng_key_, args)["gp_aggr"] #(num_samples, num_regions) <- (5,58)
   
    # Plotting 
    #plot_process(agg_gp_draws)
    
    #-------------------------- Initiate Training Loop -------------------------- #
    adam = numpyro.optim.Adam(step_size = args["learning_rate"])
    svi = SVI(
        vae_model, 
        vae_guide, 
        adam, 
        RenyiELBO(), 
        hidden_dim = args["hidden_dim"], 
        z_dim = args["z_dim"]
    )

    rng_key, rng_key_samp, rng_key_init = random.split(args["rng_key"],3)
    #(num_samples, num_regions) 
    init_batch = agg_gp_predictive(rng_key_, args)["gp_aggr"] #(num_samples, num_regions) <- i.e (5,58)
    svi_state = svi.init(rng_key_init, init_batch)
    
    test_loss_list = []
    for i in range(args["num_epochs"]):
        rng_key, rng_key_train, rng_key_test, rng_key_infer = random.split(rng_key, 4)
        t_start = time.time()

        num_train = 1000
        # Where forward/backward pass gets called for train
        train_loss , svi_state = epoch_train(rng_key_train, svi_state, num_train)

        num_test = 1000 
        # Where forward/backward pass gets called for test
        test_loss = eval_test(rng_key_test, svi_state, num_test)
        test_loss_list += [test_loss]

        print("Epoch : {}, train loss : {:.2f}, test loss : {:.2f} ({:.2f} s.)".format(i, train_loss, test_loss, time.time() - t_start))

        if math.isnan(test_loss):
            break 

    # save decoder
    decoder_params = svi.get_params(svi_state)
    save_path = f"model_weights/aggVAE_Dec_e{args['num_epochs']}_h{args['hidden_dim']}_z{args['z_dim']}"
    with open(save_path, "wb") as file:
        pickle.dump(decoder_params, file)

    



    
     
    

