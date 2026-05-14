# -------------------------------- aggVAE_prev ------------------------------- #
# Leverages VAE that learn aggregated GP's to model prevalence directly.
import os 
import pickle 
import numpy as np
import pandas as pd 
import geopandas as gpd 
from construct_spatial_matrix import compute_pts_in_polys

import jax 
import jax.numpy as jnp 
import jax.random as random 
from jax.random import PRNGKey
import numpyro 
import numpyro.distributions as dist 
from numpyro.infer import Predictive, MCMC, NUTS 

from aggVAE import Decoder
import argparse
import json
import pickle

def prev_vaeagg_betabin(
    M_lo, # polygon aggregating matrix census
    M_hi, # polygon aggregating matrix state
    n_pos_lo, # tested positive census
    n_tst_lo, # tested cases census
    n_tst_hi, # tested cases state
    lo_rgn_idx, # region index census 0..9
    hi_rgn_idx, # region index state 9..56
    dec_params, # decoder params
    pop = None
):
    z_dim, h_dim = dec_params["Dense_0"]["kernel"].shape 
    n_regions = M_lo.shape[0] + M_hi.shape[0]
    decoder = Decoder(
        hidden_dim = h_dim,
        out_dim = n_regions 
    )
    # Get N(0,1) distribution of dim z 
    z = numpyro.sample(
        "z",
        dist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim))
    ) # (*, 40)
    # Reconstruct GPs
    vae_gp_aggr = numpyro.deterministic(
        "vae_gp_aggr",
        decoder.apply({"params":dec_params},z)
    ) # (*, 55)
    # Model average influenza rate
    mu_infz = numpyro.sample("mu_infz", dist.Normal(0,1)) #(,)
    # log prevalence
    if pop is not None:
        beta_pop = numpyro.sample("beta_pop", dist.Normal(0,1))
        logit_prev = mu_infz + vae_gp_aggr + beta_pop * pop 
    else:
        logit_prev = mu_infz + vae_gp_aggr # (*, 55)
    # prevalence 
    prev = numpyro.deterministic("theta", jax.nn.sigmoid(logit_prev)) #(*,55)
    
    # Expand prevalence to the number of test cases
    prev_lo_expd = prev[lo_rgn_idx] #(*,45)
    prev_hi_expd = prev[hi_rgn_idx] #(*,227)
    prev_expd = jnp.concatenate([prev_lo_expd,prev_hi_expd])

    # Beta Binomial parameters
    concentration = numpyro.sample("concentration", dist.Gamma(2,1))
    alpha = prev_expd * concentration
    beta = (1 - prev_expd) * concentration

    # We model n_pos_hi with NaNs - We are not showing the model any
    # n_pos_hi values (number of positive cases for high resolution regions)
    n_pos_hi = jnp.repeat(jnp.nan, len(prev_hi_expd)) #(227,)
    n_pos = jnp.concatenate([n_pos_lo,n_pos_hi]) #(*,272)
    region_mask = ~jnp.isnan(n_pos)
    n_tst = jnp.concatenate([n_tst_lo, n_tst_hi]) # (*,272)

    with numpyro.handlers.mask(mask = region_mask):
        n_pos_obs = numpyro.sample(
            "n_pos_obs",
            dist.BetaBinomial(
                total_count = n_tst,
                concentration1 = alpha,
                concentration0 = beta
            ),
            obs = n_pos.astype(jnp.float32)
        )
    return n_pos_obs

if __name__ == "__main__":
    # ----------------------------------- Args ----------------------------------- #
    parser = argparse.ArgumentParser(description="Run aggVAE Prevalence Model")
    parser.add_argument("--data_root",type = str, default="data/processed/v4_clin_labs")
    parser.add_argument("--vae_params",type = str, default = "model_runs/py/v4/c9s46s3077/aggvae_ep20_sb10k_nopopw_fxd")
    parser.add_argument("--use_pop", action = "store_true", help = "add log population as covariate to model")
    parser.add_argument("--num_warmup", type = int, default=500, help = "number of warmup samples")
    parser.add_argument("--num_samples", type = int, default=1000, help = "number of mcmc samples")
    parser.add_argument("--save_folder", type = str, default = "output")
    args = parser.parse_args()

    # --------------------------------- Load data -------------------------------- #
    geo_folder = "census9_state46_county3077"
    gdf_census = gpd.read_file(os.path.join(args.data_root, geo_folder, "census9","census9.shp"))
    gdf_state = gpd.read_file(os.path.join(args.data_root, geo_folder,"state46", "state46.shp"))
    gdf_county = gpd.read_file(os.path.join(args.data_root,geo_folder,"county3077","county3077.shp"))
    # Construct Region indexes - These are needed within numpyro model to identify regions
    # and their influenza estimates
    census_rgn_idx = {r:i for i,r in enumerate(gdf_census.region)}
    state_rgn_idx = {r:i+9 for i,r in enumerate(gdf_state.region)}
    gdf_census["r_idx"] = gdf_census["region"].map(census_rgn_idx)
    gdf_state["r_idx"] = gdf_state["region"].map(state_rgn_idx)
    # Load Influenza data
    df_infz_census = pd.read_csv(os.path.join(args.data_root,"census_cln_labs9.csv"))
    df_infz_state = pd.read_csv(os.path.join(args.data_root,"state_cln_labs47.csv"))
    # Remove New Jersey - No influenza data for this region
    df_infz_state = df_infz_state[df_infz_state.region != "New Jersey"]
    # Ensure all regions are consistant
    assert len(np.setdiff1d(df_infz_census.region.unique(), gdf_census.region)) == 0
    assert len(np.setdiff1d(gdf_census.region, df_infz_census.region.unique())) == 0
    assert len(np.setdiff1d(df_infz_state.region.unique(), gdf_state.region)) == 0
    assert len(np.setdiff1d(gdf_state.region, df_infz_state.region.unique())) == 0
    
    # ------------------------------ Data Processing ----------------------------- #
    # Remove data after 2019 as covid period has widely varying data
    df_infz_census = df_infz_census[df_infz_census["year"] <= 2019]
    df_infz_state = df_infz_state[df_infz_state["year"] <= 2019]
    # Yearly data
    #todo : we need to be able to use weekly data for temporal modelling
    # Group and get yearly sum for n_tested, n_specimens
    df_infz_census_yr = df_infz_census.groupby(["region","year"])[["total_specimens","total_cases"]].agg(np.sum).reset_index()
    df_infz_state_yr = df_infz_state.groupby(["region","year"])[["total_specimens","total_cases"]].agg(np.sum).reset_index()

    # Merge Geodataframe with influenza data
    gdf_lo_yr = pd.merge(gdf_census, df_infz_census_yr, on = "region", how = "left")
    gdf_hi_yr = pd.merge(gdf_state, df_infz_state_yr, on = "region", how = "left")

    # Filter only columns required
    gdf_lo_yr = gdf_lo_yr.filter(["region","r_idx","year","total_specimens", "total_cases","geometry"])
    gdf_hi_yr = gdf_hi_yr.filter(["region","r_idx","year","total_specimens","total_cases","geometry"])

    # County level grid
    coords = gdf_county.filter(items = ["centroid_x","centroid_y"])     
    pol_pts_lo, pt_which_pol_lo = compute_pts_in_polys(coords,gdf_census)
    pol_pts_hi, pt_which_pol_hi = compute_pts_in_polys(coords,gdf_state)

    # Population
    gdf_county["census_idx"] = pt_which_pol_lo
    gdf_county["state_idx"] = pt_which_pol_hi + 9 # we need to add 9 to ensure it goes from 9..55
    census_pop = gdf_county.groupby("census_idx")["pop2020"].agg(np.sum).astype(np.float32)
    state_pop = gdf_county.groupby("state_idx")["pop2020"].agg(np.sum).astype(np.float32)
    pop = np.hstack([census_pop.values, state_pop.values])
    logpop = np.log(pop)

    # ---------------------------- Load Decoder params --------------------------- #
    
    with open(os.path.join(args.vae_params, "dec_wts"), "rb") as f:
        vae_params = pickle.load(f)
    dec_params = vae_params["decoder$params"]
        
    # --------------------- Run BetaBinomial Prevalence model -------------------- #
    mcmc = MCMC(
        NUTS(prev_vaeagg_betabin),
        num_warmup = args.num_warmup,
        num_samples = args.num_samples
    )
    mcmc.run(
        PRNGKey(0),
        M_lo = pol_pts_lo,
        M_hi = pol_pts_hi,
        n_pos_lo = jnp.array(gdf_lo_yr.total_cases.values),
        n_tst_lo = jnp.array(gdf_lo_yr.total_specimens.values),
        n_tst_hi = jnp.array(gdf_hi_yr.total_specimens.values),
        lo_rgn_idx = jnp.array(gdf_lo_yr.r_idx.values),
        hi_rgn_idx = jnp.array(gdf_hi_yr.r_idx.values),
        dec_params = dec_params,
        pop = jnp.array(logpop) if args.use_pop else None
    )
    # ----------------------------- Posterior samples ---------------------------- #
    pos_samples = mcmc.get_samples()
    mcmc.print_summary(exclude_deterministic=False)

    save_name = os.path.basename(args.vae_params)
    save_path = os.path.join(args.save_folder, save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path,"pos_samples.pkl"),"wb") as f:
        pickle.dump(pos_samples, f)

    info = {
        "num_samples" : args.num_samples,
        "num_warmup" : args.num_warmup,
        "popw" : args.use_pop,
        "decoder" : args.vae_params
    }

    with open(os.path.join(save_path,"model_info.json"),"w") as f:
        json.dump(info, f)


    
