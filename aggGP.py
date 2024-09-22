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
def case_est_model_gp_aggr(args, y = None):
    """Aggregated Gaussian Process model"""
    #n_cases = args["n_low_obs"] # (8,)
    n_specimens = args["n_specimens"]
    #? we dont need high resolution data as we are estimating it
    #n_hi_obs = args["n_high_obs"] #(48,)
    x = args["x"]
    gp_kernel = args["gp_kernel"]
    noise = args["noise"]
    jitter = args["jitter"]
    M = args["M"]
    #? we dont need high resolution data as we are estimating it
    #M2 = args["pol_pt_hi"]

    # random effects - aggregated GP 
    # Hyperparamas for kernel covariance
    length = numpyro.sample("kernel_length", dist.InverseGamma(3,3))
    var = numpyro.sample("kernel_var", dist.HalfNormal(0.05))
    # GP Kernel
    k = gp_kernel(x,x,var, length, noise, jitter)
    # GP Draw
    f = numpyro.sample(
        "f", 
        dist.MultivariateNormal(
            loc = jnp.zeros(x.shape[0]), 
            covariance_matrix= k)
        )
    
    # aggregated f into gp_aggr according to indexing of (point in polygon)
    gp_aggr = numpyro.deterministic("gp_aggr", M_g(M, f))
    
    # fixed effects
    b0 = numpyro.sample("b0", dist.Normal(0,1))
    # Linear predictor
    lp = b0 + gp_aggr 
    theta = numpyro.deterministic("theta", jax.nn.sigmoid(lp))

    #todo : Check with Swapnil if "n_specimens" is the right argument to pass to "total_count"
    numpyro.sample(
        "cases", 
        dist.BinomialLogits(
            total_count= n_specimens, 
            logits = lp
        ), 
        obs = y
    )


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
    
    # ---------------------------- Variables for Model ---------------------------- #
    args = {
        #"n_low_obs" : df_lo.tot_cases, #observarions <- This is passed as y
        "n_specimens" : jnp.array(df_lo.tot_specs),  
        "x" : jnp.array(x),
        "gp_kernel" : exp_sq_kernel,
        "jitter" : 1e-4,
        "noise" : 1e-4,
        "M" : pol_pt_lo,
        #todo : Check with Swapnil, do we need M2 : pol_pt_hi
    }

    # ---------------------------- Aggregated GP model --------------------------- #
    run_key, predict_key = random.split(random.PRNGKey(3))
    n_warm = 200
    n_samples = 1000
    mcmc = MCMC(
        NUTS(case_est_model_gp_aggr), 
        num_warmup=n_warm,
        num_samples = n_samples
    )

    # --------------------------------- Run MCMC --------------------------------- #
    start = time.time()
    mcmc.run(run_key, args, y = np.array(df_lo.tot_cases))
    end = time.time()
    t_elapsed_min = round((end - start)/60) 
    print(f"Time taken for aggGP : {t_elapsed_min}'min")

    # -------------------------------- Save Model -------------------------------- #
    with open(f"model_weights/aggGP_samples{n_samples}_tt{t_elapsed_min}min", 'wb') as file:
        dill.dump(mcmc, file)

    print("\nMCMC elapsed time:", round(end), "s")
    print("\nMCMC elapsed time:", round(end/60), "min")
    print("\nMCMC elapsed time:", round(end/(60*60)), "h")