import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import jax 
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Float, Array
import numpyro 
import numpyro.distributions as dist 
from numpyro.infer import MCMC, NUTS, Predictive
from sklearn.preprocessing import MinMaxScaler
from utils import get_comp_grid
from aggVAEv2 import decoder

import time
import pickle
import argparse

def prev_model_vae_aggr_betabin(
    x:Float[Array, "n_grid_pts 2"],
    n_tested_lo:Float[Array, "n_regions_lo"],
    n_tested_hi:Float[Array, "n_regions_hi"],
    pop_lo:Float[Array, "n_regions_lo"],
    pop_hi:Float[Array, "n_regions_hi"],
    n_positive_lo:Float[Array, "n_regions_lo"],
    M_lo:Float[Array, "n_regions_lo n_grid_pts"],
    M_hi:Float[Array, "n_regions_hi n_grid_pts"],
    decoder_params:list,
    noise:float = 1e-4,
    jitter:float = 1e-4,
    inference = False
):
    """ 
    Aggregated Gaussian Process VAE model 
    Trained decoder used to construct low dimension GP which will be 
    used to estimate prevalence using beta binomial distribution 
    Inputs 
    - x: Lat/Lon coordinates of grid points
    - n_tested_lo: Number of RT tests for low resolution regions
    - n_tested_hi: Number of RT tests for high resolution regions
    - M_lo: Low resolution region matrix (n_regions_lo x n_grid_pts)
    - M_hi: High resolution region matrix (n_regions_hi x n_grid_pts)
    - n_positive_lo: Number of positive cases for low resolution regions
    - noise: Noise level for the GP
    - jitter: Jitter for numerical stability
    - inference : whether we are running MCMC or not
    """
    # Construct f_gp with Normal(0,1) using decoder

    # Hidden side of trained decoder
    z_dim, h_dim = decoder_params[0][0].shape  #40, 50
    # Output dimensions 
    regions = M_lo.shape[0] + M_hi.shape[0] # 58
    z = numpyro.sample(
        "z",
        dist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim)) 
    ) #(40,)
    dec_init_fn, dec_apply_fn = decoder(h_dim, regions)
    vae_gp_aggr = numpyro.deterministic(
        "vae_gp_aggr",
        dec_apply_fn(decoder_params, z)
    )

    pop = jnp.concatenate([pop_lo, pop_hi],axis = 0) #(58,)
    n_tested = jnp.concatenate([n_tested_lo,n_tested_hi], axis = 0) #(58,)

    # Beta Binomial Distribution for influenza prevalence estimation

    # fixed effects 
    # captures the baseline influenza estimates
    mu_infz = numpyro.sample("mu_infz", dist.Normal(0,1))
    # random effects 
    # captures spatial correlation in latent prevelance
    f_gp_approx = vae_gp_aggr
    # covariates : fixed (This is the coefficents of population covariants)
    beta = numpyro.sample("beta", dist.Normal(0,1))
    # Logists revelence
    logits_prev = mu_infz + f_gp_approx + beta * pop
    #  prevalence values (out target,y)
    prev = numpyro.deterministic("theta", jax.nn.sigmoid(logits_prev))
    # Hyperparams for Beta-Binomial Distibution
    concentration = numpyro.sample("concentration", dist.Gamma(2,1)) #(,)
    alpha = prev * concentration #(,)
    beta = (1 - prev) * concentration #(,)

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
    return n_positive_obs

def show_mcmc_diagnostics(ss, n_regn_lo, n_regn_hi):
    try:
        r_hat_vals = ss["vae_gp_aggr"]["r_hat"]
        n_eff_vals = ss["vae_gp_aggr"]["n_eff"]

        if jnp.isnan(r_hat_vals).any() or jnp.isnan(n_eff_vals).any():
            print("Warning NaN values detected in r_hat or n_eff")

        ess_lo = np.mean(n_eff_vals[0:n_regn_lo])
        r_hat_lo = np.max(r_hat_vals[0:n_regn_lo])
        ess_hi = np.mean(n_eff_vals[n_regn_lo:n_regn_lo + n_regn_hi])
        r_hat_hi = np.max(r_hat_vals[n_regn_lo:n_regn_lo + n_regn_hi])
        print("Average ESS for all aggVAE-low effects : ", round(ess_lo,2))
        print("Max r_hat for all aggVAE-low effects : ", round(r_hat_lo,2))
        print("Average ESS for all aggVAE-high effects : ", round(ess_hi,2))
        print("Max r_hat for all aggVAE-high effects : ", round(r_hat_hi,2))
    except Exception as e:
        print("Error in diagnostic summary : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run train decoder to get low dimension GPs to compute prevalence")
    parser.add_argument("--data_path", type = str, default = "data/processed", help = "Path to directory with data and saved vectors")
    parser.add_argument("--vae_pretrain_rt", type = str, default = "model_runs/aggvae_ep20_h50_z40_fixed")
    parser.add_argument("--n_warmup", type = int, default = 500)
    parser.add_argument("--n_samples", type = int, default = 500)
    args = parser.parse_args()
    # -------------------------- Data and Saved Vectors -------------------------- #
    grid_data = get_comp_grid(Path(args.data_path))
    # Load necessay saved vectors
    x:Float[Array, "n_grid_pts 2"] = jnp.array(grid_data["x"])
    pol_pt_lo:Float[Array, "n_regions n_grid_pts"] = jnp.array(grid_data["pol_pt_lo"])
    pt_which_pol_lo:Float[Array, "n_grid_pts"] = jnp.array(grid_data["pt_which_pol_lo"])
    pol_pt_hi:Float[Array, "n_regions n_grid_pts"] = jnp.array(grid_data["pol_pt_hi"])
    pt_which_pol_hi:Float[Array, "n_grid_pts"] = jnp.array(grid_data["pt_which_pol_hi"])
    df_lo:gpd.GeoDataFrame = gpd.read_file(Path(args.data_path) / "low" / "us_census_divisions" / "us_census_divisions.shp")
    df_hi:gpd.GeoDataFrame = gpd.read_file(Path(args.data_path) / "high" / "us_state_divisions" / "us_state_divisions.shp" )
    # Influenza related data
    n_tested_lo:Float[Array, "n_regions_lo"] = jnp.array(df_lo["tot_specs"].values) #(9,) <- No of RT Tests
    n_tested_hi:Float[Array, "n_regions_hi"] = jnp.array(df_hi["tot_specs"].values) #(49,) <- No of RT Tests
    n_positive_lo:Float[Array, "n_regions_lo"] = jnp.array(df_lo["tot_cases"].values) #(9,) <- No of total positive cases
    # Note this is the ground truth, we are trying to find the total positive cases for high resolution regions
    n_positive_hi:Float[Array, "n_regions_hi"] = jnp.array(df_hi["tot_cases"].values) #(40,) <- No of total positive cases
    # Read population 
    pop_lo = pd.read_csv(Path(args.data_path) / "low" / "census_popn.csv")
    pop_hi = pd.read_csv(Path(args.data_path) / "high" / "state_popn.csv")
    scaler = MinMaxScaler()
    pop = scaler.fit_transform(np.concatenate([pop_lo["tot_popn"].values, pop_hi["tot_popn"].values]).reshape(-1,1))
    n_popn_lo:Float[Array, "n_regions_lo"] = jnp.array(pop[:pop_lo.shape[0]].ravel())
    n_popn_hi:Float[Array, "n_regions_hi"] = jnp.array(pop[pop_lo.shape[0]:].ravel())

    # --------------------------- Load Trained Decoder --------------------------- #
    with open(os.path.join(args.vae_pretrain_rt, "dec_weights"), "rb") as file:
        vae_params = pickle.load(file)
        decoder_params = vae_params["decoder$params"]
    
    # Instatiate the model
    key,subkey = random.split(random.PRNGKey(3),2)
    mcmc = MCMC(
        NUTS(prev_model_vae_aggr_betabin),
        num_warmup = args.n_warmup,
        num_samples = args.n_samples
    )
    # Run MCMC 
    start = time.time() 
    mcmc.run(
        key, 
        x = x,
        n_tested_lo =n_tested_lo,
        n_tested_hi = n_tested_hi,
        pop_lo = n_popn_lo,
        pop_hi = n_popn_hi,
        n_positive_lo = n_positive_lo,
        M_lo = pol_pt_lo,
        M_hi = pol_pt_hi,
        decoder_params = decoder_params,
        noise = 1e-4,
        jitter = 1e-4,
        inference = False 
    )
    end = time.time()

    # Get Samples
    prev_samples = mcmc.get_samples()
    mcmc.print_summary(exclude_deterministic=False)

    # Diagnostics
    ss = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))
    n_regn_lo = df_lo.shape[0]
    n_regn_hi = df_hi.shape[0]
    show_mcmc_diagnostics(ss, n_regn_lo, n_regn_hi)

    # Posterior prediction
    pos_key, subkey = random.split(subkey)
    prev_predictive_vae = Predictive(prev_model_vae_aggr_betabin, prev_samples)
    prev_pos_samples = prev_predictive_vae(
        pos_key,
        x = x,
        n_tested_lo = n_tested_lo,
        n_tested_hi = n_tested_hi, 
        pop_lo = n_popn_lo,
        pop_hi = n_popn_hi,
        n_positive_lo = None, 
        M_lo = pol_pt_lo, 
        M_hi = pol_pt_hi, 
        decoder_params = decoder_params,
        inference = True
    )

    n_positive_obs_mean = prev_pos_samples["n_positive_obs"].mean(axis = 0)
    theta_pos_mean = prev_pos_samples["theta"].mean(axis = 0)
    df_lo["prev"] = df_lo.tot_cases / df_lo.tot_specs
    df_lo["tot_cases_pred"] = n_positive_obs_mean[:n_regn_lo]
    df_lo["prev_pred"] = theta_pos_mean[:n_regn_lo]
    df_hi["prev"] = df_hi.tot_cases / df_hi.tot_specs
    df_hi["tot_cases_pred"] = n_positive_obs_mean[n_regn_lo:n_regn_lo+n_regn_hi]
    df_hi["prev_pred"] = theta_pos_mean[n_regn_lo:n_regn_lo+n_regn_hi]

    print(f"Saving results ar : {args.vae_pretrain_rt}")
    df_lo.to_csv(os.path.join(args.vae_pretrain_rt,"lo_preds.csv"), index = False)
    df_hi.to_csv(os.path.join(args.vae_pretrain_rt,"hi_preds.csv"), index = False)
