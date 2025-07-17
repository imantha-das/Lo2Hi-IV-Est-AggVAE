# ---------------------------------- Imports --------------------------------- #
from pathlib import Path, PosixPath

import numpy as np 
import geopandas as gpd

import jax 
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Float, Integer, Bool
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from typing import List, Dict, Tuple, Union

import time
import dill

from utils import get_comp_grid, M_g, exp_sq_kernel
import argparse 


# ---------------------------- Aggregate Model V1 ---------------------------- #

def prev_model_gp_aggr_bin(
    x:Float[Array, "n_grids_pts 2"],
    n_tested_lo:Float[Array, "n_regions"],
    n_tested_hi:Float[Array, "n_regions"],
    M_lo:Float[Array, "n_regions n_grid_pts"],
    M_hi:Float[Array, "n_regions n_grid_pts"],
    n_positive_lo:Union[None,Float[Array, "n_regions_lo"]],
    noise:Float = 1e-4,
    jitter:Float = 1e-4,
    inference = False
    ):
    """
    Aggregated Gaussian Process Model 
    #! model prevelnce as b0 + gp_aggr & Binomial Dist
    - x: Lat/Lon coordinates of grid points
    - n_tested_lo: Number of RT tests for low resolution regions
    - n_tested_hi: Number of RT tests for high resolution regions
    - M_lo: Low resolution region matrix (n_regions_lo x n_grid_pts)
    - M_hi: High resolution region matrix (n_regions_hi x n_grid_pts)
    - n_positive_lo: Number of positive cases for low resolution regions
    - noise: Noise level for the GP
    - jitter: Jitter for numerical stability
    - inference: If True, the model is in inference mode (no observations)
    """ 

    # Sample values from distribution for hyperparams
    kernel_length = numpyro.sample("kernel_length", dist.InverseGamma(3,3))
    kernel_var = numpyro.sample("kernel_var", dist.HalfNormal(0.05))

    # Compute Kernel
    k = exp_sq_kernel(x,x,kernel_var,kernel_length, noise, jitter) # e.g (2618,2618)
    # GP
    f = numpyro.sample(
        "f",
        dist.MultivariateNormal(loc = jnp.zeros(x.shape[0]), covariance_matrix = k),
        obs = None
    ) # e.g (2618,)

    # Aggregate for all low/high points
    gp_aggr_lo:Float[Array, "n_regions n_samples"] = numpyro.deterministic("gp_aggr_lo", M_g(M_lo,f)) #e.g. (9,) <- note this is the shape for one sample
    gp_aggr_hi:Float[Array, "n_regions n_samples"] = numpyro.deterministic("gp_aggr_hi", M_g(M_hi,f)) #e.g. (49,)
    # Concatenate low and high regions
    gp_aggr:Float[Array, "n_region_lo+n_region_hi n_samples"] = numpyro.deterministic("gp_aggr", jnp.concatenate([gp_aggr_lo,gp_aggr_hi])) #eg. (58,)

    # Fixed effects
    b0:Float = numpyro.sample("b0", dist.Normal(0,1)) # e.g. (,)
    # linear predictor = Fixed effects (b0.X) + random effects (GP)
    lp:Float[Array, "n_regions_lo+hi n_samples"] = b0 + gp_aggr # e.g. (58,)

    #Theta represents the prevalence values (our target/y) 
    theta:Float[Array, "n_regions_lo+hi n_samples"] = numpyro.deterministic("theta", jax.nn.sigmoid(lp))

    # We show all tested case values, which is n in our Binomial regression model below
    n_tested:Float[Array, "n_regions_lo+hi"] = jnp.concatenate([n_tested_lo, n_tested_hi], axis = 0) #e.g (58,)

    # We need to make tested positive cases array of shape : (n_regions_lo+hi) 
    # where hi values are NaN's. This is because we are planning to show the model
    # only low region tested cases to predict for the hi cases.
    if not inference:
        n_positive_hi:Float[Array, "n_regions_hi"] = jnp.repeat(jnp.nan, M_hi.shape[0]) #e.g. (49,)
        n_positive = jnp.concatenate([n_positive_lo, n_positive_hi], axis = 0) #e.g. (58,)
        region_mask = ~jnp.isnan(n_positive) # e.g [True x 9 ... False x 49]
        
        # We need to use numpyro.handlers.mask to make sure we can account for NaN values in observations
        with numpyro.handlers.mask(mask = region_mask):
            n_positive_obs = numpyro.sample(
                "n_positive_obs",
                dist.BinomialLogits(total_count=n_tested, logits = lp),
                obs = n_positive
            )
    else:
        # Inference mode, we do not have n_positive_obs
        n_positive_obs = numpyro.sample(
            "n_positive_obs",
            dist.BinomialLogits(total_count=n_tested, logits = lp),
            obs = None
        )
    return mcmc

# ---------------------------- Aggregate Model V2 ---------------------------- #
def prev_model_gp_aggr_betabin(
    x:Float[Array, "n_grids_pts 2"],
    n_tested_lo:Float[Array, "n_regions"],
    n_tested_hi:Float[Array, "n_regions"],
    M_lo:Float[Array, "n_regions n_grid_pts"],
    M_hi:Float[Array, "n_regions n_grid_pts"],
    n_positive_lo:Union[None,Float[Array, "n_regions_lo"]],
    noise:Float = 1e-4,
    jitter:Float = 1e-4,
    inference = False
    ):
    """
    Aggregated Gaussian Process Model
    !Modelling p as just gp_aggr + Beta Bionomial
    - x: Lat/Lon coordinates of grid points
    - n_tested_lo: Number of RT tests for low resolution regions
    - n_tested_hi: Number of RT tests for high resolution regions
    - M_lo: Low resolution region matrix (n_regions_lo x n_grid_pts)
    - M_hi: High resolution region matrix (n_regions_hi x n_grid_pts)
    - n_positive_lo: Number of positive cases for low resolution regions
    - noise: Noise level for the GP
    - jitter: Jitter for numerical stability
    - inference: If True, the model is in inference mode (no observations)
    """ 

    # Sample values from distribution for hyperparams
    kernel_length = numpyro.sample("kernel_length", dist.InverseGamma(3,3))
    kernel_var = numpyro.sample("kernel_var", dist.HalfNormal(0.05))

    # Compute Kernel
    k = exp_sq_kernel(x,x,kernel_var,kernel_length, noise, jitter) # e.g (2618,2618)
    # GP
    f = numpyro.sample(
        "f",
        dist.MultivariateNormal(loc = jnp.zeros(x.shape[0]), covariance_matrix = k),
        obs = None
    ) # e.g (2618,)

    # Aggregate for all low/high points
    gp_aggr_lo:Float[Array, "n_regions n_samples"] = numpyro.deterministic("gp_aggr_lo", M_g(M_lo,f)) #e.g. (9,) <- note this is the shape for one sample
    gp_aggr_hi:Float[Array, "n_regions n_samples"] = numpyro.deterministic("gp_aggr_hi", M_g(M_hi,f)) #e.g. (49,)
    # Concatenate low and high regions
    gp_aggr:Float[Array, "n_region_lo+n_region_hi n_samples"] = numpyro.deterministic("gp_aggr", jnp.concatenate([gp_aggr_lo,gp_aggr_hi])) #eg. (58,)

    # linear predictor = Fixed effects (b0.X) + random effects (GP)
    lp:Float[Array, "n_regions_lo+hi n_samples"] = gp_aggr # e.g. (58,)

    #Theta represents the prevalence values (our target/y) 
    theta:Float[Array, "n_regions_lo+hi n_samples"] = numpyro.deterministic("theta", jax.nn.sigmoid(lp))

    # We show all tested case values, which is n in our Binomial regression model below
    n_tested:Float[Array, "n_regions_lo+hi"] = jnp.concatenate([n_tested_lo, n_tested_hi], axis = 0) #e.g (58,)

    # Hyperparameters for Beta-Binomial distribution
    concentration = numpyro.sample("concentration", dist.Gamma(2,1)) #(,)
    alpha = theta * concentration #(,)
    beta = (1 - theta) * concentration #(,)

    # We need to make tested positive cases array of shape : (n_regions_lo+hi) 
    # where hi values are NaN's. This is because we are planning to show the model
    # only low region tested cases to predict for the hi cases.
    if not inference:
        n_positive_hi:Float[Array, "n_regions_hi"] = jnp.repeat(jnp.nan, M_hi.shape[0]) #e.g. (49,)
        n_positive = jnp.concatenate([n_positive_lo, n_positive_hi], axis = 0) #e.g. (58,)
        region_mask = ~jnp.isnan(n_positive) # e.g [True x 9 ... False x 49]
        
        # We need to use numpyro.handlers.mask to make sure we can account for NaN values in observations
        with numpyro.handlers.mask(mask = region_mask):
            n_positive_obs = numpyro.sample(
                "n_positive_obs",
                dist.BetaBinomial(total_count=n_tested, concentration1=alpha, concentration0=beta),
                obs = n_positive.astype(jnp.float32)
            )
    else:
        # Inference mode, we do not have n_positive_obs
        n_positive_obs = numpyro.sample(
            "n_positive_obs",
            dist.BetaBinomial(total_count=n_tested, concentration1=alpha, concentration0=beta),
        )
    return mcmc



# ----------------------------------- Main ----------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the aggregated GP model")
    parser.add_argument("--data_path", type = str, default ="data/processed", help="Path to the processed data directory")
    parser.add_argument("--n_warmup", type=int, default=10, help="Number of warmup steps for MCMC")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples for MCMC")
    args = parser.parse_args()
    # Lat/Lon values for artificial grid
    data_dir = Path(args.data_path)
    grid_data = get_comp_grid(data_dir)
    # Get variables from grid data
    x:Float[Array, "n_grid_pts 2"] = jnp.array(grid_data["x"])
    pol_pt_lo:Float[Array, "n_regions n_grid_pts"] = jnp.array(grid_data["pol_pt_lo"])
    pt_which_pol_lo:Float[Array, "n_grid_pts"] = jnp.array(grid_data["pt_which_pol_lo"])
    pol_pt_hi:Float[Array, "n_regions n_grid_pts"] = jnp.array(grid_data["pol_pt_hi"])
    pt_which_pol_hi:Float[Array, "n_grid_pts"] = jnp.array(grid_data["pt_which_pol_hi"])
    df_lo:gpd.GeoDataFrame = grid_data["df_lo"]
    df_hi:gpd.GeoDataFrame = grid_data["df_hi"]
    # Influenza related data
    n_tested_lo:Float[Array, "n_regions_lo"] = jnp.array(df_lo["tot_specs"].values) #(9,) <- No of RT Tests
    n_tested_hi:Float[Array, "n_regions_hi"] = jnp.array(df_hi["tot_specs"].values) #(49,) <- No of RT Tests
    n_positive_lo:Float[Array, "n_regions_lo"] = jnp.array(df_lo["tot_cases"].values) #(9,) <- No of total positive cases
    # Note this is the ground truth, we are trying to find the total positive cases
    # for high resolution regions
    n_positive_hi:Float[Array, "n_regions_hi"] = jnp.array(df_hi["tot_cases"].values) #(40,) <- No of total positive cases

    # Instatiate the model
    key,subkey = random.split(random.PRNGKey(3),2)
    mcmc = MCMC(
        NUTS(prev_model_gp_aggr_v2),
        num_warmup = args.n_warmup,
        num_samples = args.n_samples
    )

    # Run MCMC
    start = time.time()
    mcmc.run(
        key,
        x = x,
        n_tested_lo = n_tested_lo,
        n_tested_hi = n_tested_hi,
        M_lo = pol_pt_lo,
        M_hi = pol_pt_hi,
        n_positive_lo = n_positive_lo,
        noise = 1e-4,
        jitter = 1e-4,
        inference = False  # Set to True if you want to run inference without observations
    )

    end = time.time()
    t_elapsed_min = round((end-start)/60)
    print(f"Time takens for aggGP : {t_elapsed_min} minutes")

    # Save the model
    f_path = f"model_weights/aggGP_nsamples_{args.n_samples}_v5.0"
    with open(f_path, 'wb') as file:
        dill.dump(mcmc, file)

    print("\nMCMC elapsed time:", round(end), "s")
    print("\nMCMC elapsed time:", round(end/60), "min")
    print("\nMCMC elapsed time:", round(end/(60*60)), "h")