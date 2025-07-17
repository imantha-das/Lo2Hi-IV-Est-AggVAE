# ==============================================================================
# Script to Run Aggregated Gaussian Process on Influenza Cases at differnt 
# resolution administrative boundaries.
# ==============================================================================

# ---------------------------------- Imports --------------------------------- #

import os
import numpy as np 
import pandas as pd
import geopandas as gpd

import time 

import jax
import jax.numpy as jnp 
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

from termcolor import colored

import dill
# ---------------------------- GP Kernel Function ---------------------------- #
def dist_euclid(x, z):
    """
    Computes Eucledian Distance Between Regions. This function is used by
    exp_sq_kernel function (kernel function for gaussian processes)
    """
    x = jnp.array(x) # (ngrid_pts, lat/lon) <- i.e (2618,2)
    z = jnp.array(z) # (ngrid_pts, lat/lon) <- i.e (2618,2)
    if len(x.shape)==1:
        x = x.reshape(x.shape[0], 1) #(2618,) -> (2618,1)
    if len(z.shape)==1:
        z = x.reshape(x.shape[0], 1) #(2618,) -> (2618,1)
    n_x, m = x.shape # 2618 , 2
    n_z, m_z = z.shape # 2618 , 2
    assert m == m_z
    delta = jnp.zeros((n_x,n_z)) #(ngrid_pts,ngrid_pts) <- i.e (2618,2618)
    for d in jnp.arange(m):
        x_d = x[:,d] #(ngrid_pts-lat/lon,) <- (2618,) 
        z_d = z[:,d] #(ngrid_pts-lat/lon,) <- (2618,) 
        delta += (x_d[:,jnp.newaxis] - z_d)**2 # (2618,2618)

    return jnp.sqrt(delta) #(2618,2618)


def exp_sq_kernel(x, z, var, length, noise, jitter=1.0e-4):
    dist = dist_euclid(x, z) #(2618, 2618)
    deltaXsq = jnp.power(dist/ length, 2.0) 
    k = var * jnp.exp(-0.5 * deltaXsq)
    k += (noise + jitter) * jnp.eye(x.shape[0])
    return k # (ngrid_pts, ngrid_pts) <- (2618,2618)

# --------------------------- Aggregation functions -------------------------- #
def M_g(M, g):
    '''
    - $M$ is a matrix with binary entries $m_{ij},$ showing whether point $j$ is in polygon $i$
    - $g$ is a vector of GP draws over grid
    - $maltmul(M, g)$ gives a vector of sums over each polygon
    '''
    M = jnp.array(M)
    g = jnp.array(g).T
    return(jnp.matmul(M, g))

# ------------------------ Aggregated prevelance model ----------------------- #
def prev_model_gp_aggr(args, tested_positive = None):
    """Aggregated Gaussian Process model"""
    n_specimens_lo = args["n_specimens_lo"] # number of tests conducted
    n_specimens_hi = args["n_specimens_hi"]
    x = args["x"] #num grid pts for lat/lon : (2618,2)
    gp_kernel = args["gp_kernel"] # Gaussian process kernal
    noise = args["noise"] #(,)
    jitter = args["jitter"]# (,)
    M_lo = args["M_lo"] # R in [0,1] where 1 indicates point inside a region : (9, 2618)
    M_hi = args["M_hi"] # (49,2618)
    kernel_length = args["kernel_length"] #dist.InverseNormal(3,3)
    kernel_var = args["kernel_var"] #dist.HalfNormal(0.05)

    # random effects - aggregated GP 
    # Hyperparamas for kernel covariance
    length = numpyro.sample("kernel_length", kernel_length)
    var = numpyro.sample("kernel_var", kernel_var)
    # GP Kernel
    k = gp_kernel(x,x,var, length, noise, jitter)
    # GP Draw : Retuns a value for each grid point
    f = numpyro.sample(
        "f", 
        dist.MultivariateNormal(
            loc = jnp.zeros(x.shape[0]), 
            covariance_matrix= k)
        ) #(2618,)
    
    # Pick relevent points for all the low regions (n points) from all the points (2618)
    # this returns the aggregated values per region ...
    gp_aggr_lo = numpyro.deterministic("gp_aggr_lo", M_g(M_lo, f)) #(9,)
    gp_aggr_hi = numpyro.deterministic("gp_aggr_hi", M_g(M_hi, f)) #(49,)
    # Now we need to aggregate both. This step is important since even though we only
    # show the model the low resolution data, to produce high resolution data it
    # needs th GP realizations for those regions
    gp_aggr = numpyro.deterministic("gp_aggr", jnp.concatenate([gp_aggr_lo,gp_aggr_hi])) #(58,)
    
    # fixed effects : Create param
    b0 = numpyro.sample("b0", dist.Normal(0,1)) #(,)
    # Linear predictor : lp probably means logit prevelance
    lp = b0 + gp_aggr #? (58,) <- values are actually pos/negative and not between 0 and 1. Could be something wrong with the gp. but also we are taking binomial logits so potentially ok

    # theta represents the prevelence value
    theta = numpyro.deterministic("theta", jax.nn.sigmoid(lp)) #(58,)

    # Add NaN values to n_positive to accomodate unavailable data for high resolution. 
    # n_positive : [low_res_pos ... nan for high_res ...]
    tested_positive = jnp.pad(tested_positive, (0, M_hi.shape[0]),constant_values = 0.0) #[3762.  484. ... , 0,0,0]
    tested_positive = jnp.where(tested_positive == 0, jnp.nan, tested_positive)# [3762.  484. ... , nan,nan,nan]
    tested_positive_mask = ~jnp.isnan(tested_positive) # [True, True, ...., False, False, False]

    # Aggregate n_specimens lo and hi
    tested_cases = jnp.concatenate([n_specimens_lo, n_specimens_hi], axis = 0) #(58,)

    #We use numpyro.handlers.mask to make sure we can account for NaN values for observations
    with numpyro.handlers.mask(mask=tested_positive_mask): 
        n_positive_obs =  numpyro.sample(
            "n_positive_obs", 
            dist.BinomialLogits(total_count=tested_cases, logits=lp), 
            obs=tested_positive
        )

    return n_positive_obs

if __name__ == "__main__":

    # --------------------------------- Load Data -------------------------------- #
    data_dir = "data/processed"
    
    # Lat/Lon Values of artificial grid
    x = np.load(os.path.join(data_dir,"lat_lon_x.npy")) #(2618,2)
    # Low regional data
    pol_pt_lo = np.load(os.path.join(data_dir,"low","pol_pt_lo.npy")) #(9,2618)
    pt_which_pol_lo = np.load(os.path.join(data_dir,"low","pt_which_pol_lo.npy")) #(2618,)
    # High regional data 
    pol_pt_hi = np.load(os.path.join(data_dir,"high","pol_pt_hi.npy")) #(49, 2618,)
    pt_which_pol_hi = np.load(os.path.join(data_dir,"high","pt_which_pol_hi.npy")) #(2618,)
    # Dataframes
    df_lo = gpd.read_file(os.path.join(data_dir, "low", "us_census_divisions","us_census_divisions.shp"))
    df_hi = gpd.read_file(os.path.join(data_dir, "high", "us_state_divisions","us_state_divisions.shp"))
    
    #* ==========================================================================
    #* -------------------- Variables that need to be changed -------------------- #
    test_cases_lo = jnp.array(df_lo.tot_specs) #(9,)
    test_cases_hi = jnp.array(df_hi.tot_specs) #(49,)
    M1 = pol_pt_lo #(9,2618)
    M2 = pol_pt_hi #(49,2618)
    tested_positive_lo = jnp.array(df_lo.tot_cases) #(9,)
    save_hi_lo = "lo"
    #* ==========================================================================

    # -------------------------- Mask Hi Resolution data ------------------------- #
    # We need to aggregate the values for the hi resolution data for total specimens
    # we dont know the values for these in real life because we are estimating these
    # the prior should handle this from what I believe
    # So create NaN values and append them for the Hi Res Regions.

    # jnp.pad soesnt accept NaNs so use 0 as constand and use where to replace 0 with nan

    
    # ---------------------------- Variables for Model ---------------------------- #
    args = {
        #"n_low_obs" : df_lo.tot_cases, #observarions <- This is passed as y
        "n_specimens_lo" : jnp.array(test_cases_lo), # Number of tests for low resolution regions, (9,)
        "n_specimens_hi" : jnp.array(test_cases_hi),
        "x" : jnp.array(x), # Lat/lon vals of grid points, (2468,2)
        "gp_kernel" : exp_sq_kernel,
        "jitter" : 1e-4, 
        "noise" : 1e-4,
        "M_lo" : pol_pt_lo, # R in [0,1] where 1 indicates point in region, (9, 2468)
        "M_hi" : pol_pt_hi, #R in [0,1] where 1 indicates point in region , (49,2468)
        # GP Kernel Hyperparams
        "kernel_length" : dist.InverseGamma(3,3), #(,)
        "kernel_var" : dist.HalfNormal(0.05)
    }

    # ---------------------------- Aggregated GP model --------------------------- #
    run_key, predict_key = random.split(random.PRNGKey(3))
    n_warm = 500
    n_samples = 500
    mcmc = MCMC(
        NUTS(prev_model_gp_aggr), 
        num_warmup=n_warm,
        num_samples = n_samples
    )

    # --------------------------------- Run MCMC --------------------------------- #
    start = time.time()
    mcmc.run(run_key, args, tested_positive = jnp.array(tested_positive_lo))
    end = time.time()
    t_elapsed_min = round((end - start)/60) 
    print(f"Time taken for aggGP : {t_elapsed_min}'min")

    # -------------------------------- Save Model -------------------------------- #
    f_path = f"model_weights/aggGP_{save_hi_lo}_nsamples_{n_samples}_tt{t_elapsed_min}min"
    with open(f_path, 'wb') as file:
        dill.dump(mcmc, file)

    print("\nMCMC elapsed time:", round(end), "s")
    print("\nMCMC elapsed time:", round(end/60), "min")
    print("\nMCMC elapsed time:", round(end/(60*60)), "h")

    