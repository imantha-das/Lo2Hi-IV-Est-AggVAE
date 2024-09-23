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

#? ==============================================================================
# Probelms when training
# VAE loss fluctuates a lot, is it an issue with data ...

# Questions
# Similar issue to aggGP. Now we have trained a decoder with z = 3 (had to make it
# small as low resolution data only has 9 regions). Now once the decoder is trained
# how to get predictions for high resolution. Pass hi resolution data though encoder
# (untrained) ... , the issue in the aggVAE problem, the authors have aggrgated
# both old and new data and then use it for prediction. But now since we are using 
# low to high this method cannot be applied as we are only training on low data.

# To make more sense there is 9 regions in the low resolutiion data
# We train E(*, 9) -> Hidden(*, 6) -> (*, 3) = z -> ...
# ... sampling from univariant -> D(z) -> Hidden(*, 6) -> (*, 9)
# Save decoder ... But now the the encoder needs (*, 49) and the decoder
# needs to produce (*, 49) when it was trained to produce (*, 9)
# So we need to access lat/lon points and sample from z so we can get (*,49) ...
# ... like how we do on images ?
#? ==============================================================================

# ------------------- Func for Prior Predictive Simulation ------------------- #
def gp_aggr(args):
    x = args["x"] # (num_grid_pts, lat+lon) <- (2618,2)
    gp_kernel = args["gp_kernel"]
    noise = args["noise"]
    
    M= args["M"] #M_lo/M_hi , i.e M_lo : (9, 2618)

    # Random effect - aggregated GP 
    length = numpyro.sample("kernel_length", dist.InverseGamma(3,3)) #(,)
    var = numpyro.sample("kernel_var",dist.HalfNormal(0.05)) #(,)
    k = gp_kernel(x,x,var, length, noise) #(num_grig_pts,num_grid_pts) <- (2618,2618)
    f = numpyro.sample(
        "f", 
        dist.MultivariateNormal(loc = jnp.zeros(x.shape[0]), covariance_matrix = k)
        ) #(num_grid_pts,) <- i.e (2618,)

    #aggregate f into gp_aggr according to indexing of (point in polygon)
    gp_aggr = numpyro.deterministic("gp_aggr", M_g(M, f)) #(num_regions,) <- i.e (9,) for lo
    return gp_aggr

    
# -------------------------- Variational Autoencoder ------------------------- #
def vae_encoder(hidden_dim = 50, z_dim = 40):
    return stax.serial(
        #todo : This will increase dimensions rather than dicrease as we have only 9 regions
        #(num_samples, num_regions) -> (num_samples, hidden_dims) 
        stax.Dense(hidden_dim, W_init = stax.randn()), #i.e(5,9) -> (5,50)
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
        # (num_samples, hidden_dim) -> (num_samples, num_regions) : (5,50) -> (5, 9)
        stax.Dense(out_dim, W_init = stax.randn())
    )


def vae_model(batch, hidden_dim, z_dim):
    """This computes the decoder portion"""
    batch = jnp.reshape(batch, (batch.shape[0], -1)) # still gonna be (5,9) for lo
    batch_dim, out_dim = jnp.shape(batch) # 5 , 116 for lo

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
    gen_loc = decode(z) #(num_regions,) : (9,)
    #(num_samples, num_regions) : (5,9)
    obs = numpyro.sample("obs", dist.Normal(gen_loc, args["vae_var"]), obs = batch) 
    return obs


def vae_guide(batch, hidden_dim, z_dim):
    """This computes the encoder portion"""
    batch = jnp.reshape(batch, (batch.shape[0], -1)) #(num_samples, num_regions) : (5,9) for lo
    batch_dim, input_dim = jnp.shape(batch)# num_samples , num_regions : 5 , 9 

    # vae-encoder in numpyro module
    encode = numpyro.module(
        name = "encoder", 
        nn = vae_encoder(hidden_dim=hidden_dim,z_dim = z_dim),
        input_shape = (batch_dim, input_dim) #(5,9)
    ) #(num_samples, num_regions) -> (num_samples, hidden_dims) : i.e (5,9) -> (5,40)

    # Samapling mu, sigma - Pretty much the forward pass
    z_loc, z_std = encode(batch) #mu : (num_samples, z_dim), sigma2 : (num_samples, z_dim) <- (5,40),(5,40)

    z = numpyro.sample("z", dist.Normal(z_loc, z_std)) #(num_sample, z_dim) : (5,40)
    return z

@jit 
def epoch_train(rng_key, svi_state, num_train):
    def body_fn(i, val):
        # Random keys
        rng_key_i = random.fold_in(rng_key, i)
        rng_key_i, rng_key_ls, rng_key_var, rng_key_noise = random.split(rng_key_i, 4)
        loss_sum, svi_state = val 
        # Gp Draw : (num_samples, num_regions)
        batch = agg_gp_predictive(rng_key, args)["gp_aggr"] # (5,9)
        print(colored(batch.shape,"green"))
        # Forward pass, Takes "state" & "args" where args are your "model inputs" 
        svi_state, loss = svi.update(svi_state, batch)
        loss_sum += loss / args["batch_size"]
        return loss_sum, svi_state 
    
    return lax.fori_loop(lower =0, upper = num_train, body_fun = body_fn, init_val = (0.0, svi_state))

@jit
def eval_test(rng_key, svi_state, num_test):
    def body_fn(i, loss_sum):
        rng_key_i = random.fold_in(rng_key, i)
        rng_key_i, rng_key_ls, rng_key_var, rng_key_noise = random.split(rng_key_i, 4)
        # GP Draw : (num_samples, num_regions) 
        batch = agg_gp_predictive(rng_key_i, args)["gp_aggr"] # (5,49)
        loss = svi.evaluate(svi_state, batch)
        loss_sum += loss 
        return loss 
    loss = lax.fori_loop(0, num_test, body_fn, 0.0)
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

    # --------------------------------- Arguments -------------------------------- #
    args = {
        "x": x,
        "gp_kernel": exp_sq_kernel,
        "noise": 1e-4,
        "M": pol_pt_lo,
        "data_used" : "lo",

        # VAE training
        "rng_key": random.PRNGKey(5),
        "num_epochs": 2, 
        #"learning_rate": 1.0e-3, 
        "learning_rate": 0.0005, 
        "batch_size": 100, 
        "hidden_dim": 6, 
        "z_dim": 3, 
        "num_train": 100,
        "num_test":100,
        "vae_var": 1,

        # Consider One Region or Both
        "consider_m1" : False
    }

    # ------------------------ Prior Predictive Simulation ----------------------- #
    rng_key, rng_key_ = random.split(random.PRNGKey(4))
    agg_gp_predictive = Predictive(gp_aggr,num_samples = 5)
    # Returns (n_Samples, num_regions_in_m1)
    # (num_sample, num_regions)
    agg_gp_draws = agg_gp_predictive(rng_key_, args)["gp_aggr"] #(5,9)
   
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
    init_batch = agg_gp_predictive(rng_key_, args)["gp_aggr"] #(5,9)
    svi_state = svi.init(rng_key_init, init_batch)
    
    test_loss_list = []
    for i in range(args["num_epochs"]):
        rng_key, rng_key_train, rng_key_test, rng_key_infer = random.split(rng_key, 4)
        t_start = time.time()

        num_train = 1000
        # Where forward/backward pass gets called for train
        _ , svi_state = epoch_train(rng_key_train, svi_state, num_train)

        num_test = 1000 
        # Where forward/backward pass gets called for test
        test_loss = eval_test(rng_key_test, svi_state, num_test)
        test_loss_list += [test_loss]

        print("Epoch : {}, train loss : {}, test loss : {} ({:.2f} s.)".format(i, test_loss, time.time() - t_start))

        if math.isnan(test_loss):
            break 

    # save decoder
    decoder_params = svi.get_params(svi_state)
    save_path = f"model_weights/aggVAE_Dec_{args['data_used']}_h{args['hidden_dim']}_z{args['z_dim']}"
    with open(save_path, "wb") as file:
        pickle.dump(file)

    



    
     
    

