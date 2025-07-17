# ==============================================================================
# Prevelance Model VAE
# ==============================================================================
import pickle
import os

import numpy as np 
import geopandas as gpd
import jax
import jax.numpy as jnp
from jax import random

import numpyro 
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import numpy as np
from aggGP import exp_sq_kernel
from aggVAE import vae_decoder

import time

def prev_model_vae_aggr(args, positive_cases = None):

    x = args["x"] #(2618,2)
    total_tests = args["n_specimens"] #num_tests : ()
    out_dims = args["out_dims"]
    
    # random effect
    decoder_params =args["decoder_params"]
    z_dim, hidden_dim = decoder_params[0][0].shape #(3, 6) 
    z = numpyro.sample("z", dist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim))) #(3,)
    _, decoder_apply = vae_decoder(hidden_dim, out_dims) # Instantiate decoder
    vae_aggr = numpyro.deterministic("vae_aggr", decoder_apply(decoder_params, z)) #(9,)
    s = numpyro.sample("sigma", dist.HalfNormal(50)) #(,)
    vae = numpyro.deterministic("vae", s * vae_aggr) #(9,)

    # Fixed effects 
    b0 = numpyro.sample("b0", dist.Normal(0,1)) #(,)
    lp = b0 + vae #(9,)
    theta = numpyro.deterministic("theta", jax.nn.sigmoid(lp)) #(9,)

    numpyro.sample(
        "positive_cases", 
        dist.BinomialLogits(total_count = total_tests, logits = lp), 
        obs = positive_cases
    ) #(9,)


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
    #* -------------------- Variables that need to be changed -------------------- #

    total_specimens = df_lo.tot_specs #(9,)
    M = pol_pt_lo #(9,2618)
    total_cases = df_lo.tot_specs #(9,)
    positive_cases = df_lo.tot_cases #(9,)
    save_hi_lo = "lo"
    out_dims = df_lo.shape[0]
    #* ==========================================================================

    # ---------------------------- Arguments to Model ---------------------------- #
    args = {
        #"n_low_obs" : df_lo.tot_cases, #observarions <- This is passed as y
        "n_specimens" : jnp.array(total_specimens),  
        "x" : jnp.array(x),
        "gp_kernel" : exp_sq_kernel,
        "jitter" : 1e-4,
        "noise" : 1e-4,
        "M" : M,
        #todo : Check with Swapnil, do we need M2 : pol_pt_hi
        # VAE training
        "rng_key": random.PRNGKey(5),
        "num_epochs": 20, 
        #"learning_rate": 1.0e-3, 
        "learning_rate": 0.0005, 
        "batch_size": 100, 
        "hidden_dim": 6, 
        "z_dim": 3, 
        "out_dims" : out_dims,
        "num_train": 100,
        "num_test":100,
        "vae_var": 1,
    }

    # ---------------------------- Load Decoder Model ---------------------------- #
    with open("model_weights/aggVAE_Dec_lo_h6_z3", "rb") as file:
        vae_params = pickle.load(file)

    encoder_params = vae_params["encoder$params"]
    decoder_params = vae_params["decoder$params"]
    args["decoder_params"] = decoder_params

    # --------------------------------- Run MCMC --------------------------------- #

    mcmc_key, predict_key = random.split(random.PRNGKey(0))
    start_time = time.time()
    mcmc = MCMC(
        NUTS(prev_model_vae_aggr),
        num_warmup = 200,
        num_samples = 1000
    )

    mcmc.run(mcmc_key, args, jnp.array(positive_cases))
    t_elapsed = time.time() - start_time
    t_elapsed_mins = int(t_elapsed / 60)

    mcmc.print_summary(exclude_deterministic = False)