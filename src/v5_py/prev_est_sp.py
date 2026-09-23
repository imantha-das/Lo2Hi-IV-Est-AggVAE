# ==============================================================================
# Prevelance Model at Spatial only
# ==============================================================================
import os 
import numpy as np
import pandas as pd 
import geopandas as gpd

import jax 
import jax.numpy as jnp 
import jax.random as random 
import numpyro 
import numpyro.distributions as dist 
from numpyro.infer import Predictive, MCMC, NUTS 

from gp_vae import Decoder
from utils import get_date_from_week, compute_pts_in_polys, M_g
import json

import plotly.express as px 
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

import pickle
from termcolor import colored

def plot_region(df,rgn_idx,y_name):
    df_n = df[df.r_idx == rgn_idx]
    reg_name = df_n.region.unique()[0]
    fig = make_subplots(rows = 1, cols = 2)
    fig.add_trace(go.Scatter(
        x = df_n.date, 
        y = df_n[y_name],
        mode = "markers",
    ), row = 1, col = 1)

    fig.add_trace(go.Histogram(
        x = df_n[y_name]
    ),row = 1, col = 2)

    fig.show()

def plot_grid(gdf_cty, idx_col):
    gdf_points = gpd.GeoDataFrame(
        gdf_cty,
        geometry=gpd.points_from_xy(gdf_cty["centroid_x"],gdf_cty["centroid_y"]),
        crs = "EPSG:4326"
    )
    fig, ax = plt.subplots(figsize = (10,8))
    gdf_points.plot(
        column = idx_col, # census_idx 
        categorical = True,
        cmap = "Set1",
        markersize = 10,
        legend = True,
        legend_kwds = {
            "title" : "Census/State Idx",
            "bbox_to_anchor" : (1,1),
            "loc" : "upper left"
        },
        ax = ax
    )
    ax.set_title("County Centroid by Census/State idxs")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.grid(True, linestyle = "--", alpha = 0.5)

def plot_true_est_prev(df_infz, pos_samples):
    """
    Plot actual and estimated prevalence as a bar graph
    @param df_infz : infuenza data at census or state level 
    @param pos_samples : posterior samples
    """
    theta_mean = pos_samples["theta"].mean(axis = 0)
    df_infz["prev"] = df_infz["total_cases"] / df_infz["total_specimens"]
    df_infz_mean = df_infz.groupby("region").agg({"prev" : "mean"})
    if len(df_infz_mean.region.unique()) == 9:
        # dataframe is at census level
        df_infz_mean["est_prev"] = theta_mean[:9]
    else:
        df_infz_mean["est_prev"] = theta_mean[9:]
    df_infz_mean.reset_index(inplace = True)
    fig = make_subplots(specs = [[{"secondary_y" : True}]])
    fig.add_trace(go.Bar(
        x = df_infz_mean["region"], 
        y = df_infz_mean["prev"], 
        name = "prev", 
        marker_color = "blue",
        opacity = 0.6,
        offsetgroup=1
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x = df_infz_mean["region"], 
        y = df_infz_mean["est_prev"], 
        name = "est prev", 
        marker_color = "orange",
        opacity = 0.6,
        offsetgroup=2
    ), secondary_y=True)
    fig.update_layout(
        title_text = "Compatison of actual and estimate mean prevalence",
        barmode = "group"
    )
    fig.show()

def prev_vae_sp_clim_county(
    dec_params,
    x_sph, # specific humidity
    x_tmmn, # minimum temperature
    M_lo, # Matrix for Low regions
    M_hi, # Matrix for high regions
    lo_rgn_idx, # region index for census 1..9
    hi_rgn_idx, # region index for states 9..55
    n_tst_lo, # Number of tests for low regions 
    n_tst_hi, # Number of tests for high regions 
    n_pos_lo # Number of positive cases for low regions
):
    """ 
    Prevelance Model 
    - Linear predictor (logit_prev) at county level 
    - Aggregate performed on linear predictor followed by applying nonlinearity (sigmoid)
    - Samples  
    """
    # Instantiate decoder
    z_dim, h2_dim = dec_params["Dense_0"]["kernel"].shape # (384, 769)
    h1_dim, out_dim = dec_params["Dense_2"]["kernel"].shape #(1558,3077)
    params = {
        "input_dims" : out_dim,
        "h1_dims" : h1_dim,
        "h2_dims" : h2_dim
    }
    decoder = Decoder(params = params)
    # Generate MVN(0,1) distirbution
    z = numpyro.sample(
        "z",
        dist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim))
    ) #(*,384)
    # Generate spatial field from trained decoder 
    vae_gp = numpyro.deterministic(
        "vae_gp",
        decoder.apply({"params" : dec_params}, z)
    ) #(*,3077)
    # Linear Predictor 
    mu_infz = numpyro.sample("mu_infz", dist.Normal(0,1)) # (,)
    beta_sph = numpyro.sample("beta_sph", dist.Normal(0,1))
    beta_tmmn = numpyro.sample("beta_tmmn", dist.Normal(0,1))
    logit_prev = mu_infz + beta_sph * x_sph + beta_tmmn * x_tmmn + vae_gp #(*,3077)

    # Aggregate 
    logit_prev_lo = M_g(M_lo, logit_prev) #(9,)
    logit_prev_hi = M_g(M_hi, logit_prev) #(46,)
    logit_prev_agg = jnp.concatenate([logit_prev_lo, logit_prev_hi])

    prev = numpyro.deterministic("theta", jax.nn.sigmoid(logit_prev_agg))

    # Expand prevalence to the number of test cases 
    prev_lo_expd = prev[lo_rgn_idx] # df_infz_lo rows
    prev_hi_expd = prev[hi_rgn_idx] # df_infz_hi rows
    prev_expd = jnp.concatenate([prev_lo_expd, prev_hi_expd])

    #! 
    eps = 1e-6
    prev_expd = jnp.clip(prev_expd, eps, 1 - eps)

    # Beta Binomial Parameters
    concentration = numpyro.sample("concentration", dist.Gamma(2,1))
    alpha = prev_expd * concentration 
    beta = (1 - prev_expd) * concentration 

    # We model high resolution positvie cases as NaNs
    n_pos_hi = jnp.repeat(jnp.nan, len(prev_hi_expd))
    n_pos = jnp.concatenate([n_pos_lo, n_pos_hi])
    region_mask = ~jnp.isnan(n_pos)
    n_tst = jnp.concatenate([n_tst_lo, n_tst_hi])

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

#todo : Aggregate GP and then apply linear predictor at census and state level
def prev_vae_sp_clim_census_state():
    pass

# =========================================================================
#  Main
# =========================================================================
if __name__ == "__main__":
    data_root = "data/processed/v5"

    gdf_census = gpd.read_file(os.path.join(data_root,"census9","census9.shp"))
    gdf_state = gpd.read_file(os.path.join(data_root,"state46","state46.shp"))
    gdf_county = gpd.read_file(os.path.join(data_root,"county3077_uniq3060","county_3077.shp"))

    df_infz_census = pd.read_csv(os.path.join(data_root, "census_cln_labs9.csv"))
    df_infz_state = pd.read_csv(os.path.join(data_root,"state_cln_labs46.csv"))
    df_infz_census = get_date_from_week(df_infz_census)
    df_infz_state = get_date_from_week(df_infz_state)

    #todo need to sort how geoid is saved - its saved as int63 but you need it in string format
    df_clim = pd.read_csv(os.path.join(data_root,"climate_county3060_wkly.csv"), dtype = {"geoid" : str})
    df_clim["date"] = pd.to_datetime(df_clim["date"])
    # Region index to ensure regions between gdf and df_infz match
    census_rgn_idx = {r:i for i,r in enumerate(gdf_census.region)}
    state_rgn_idx = {r:i+9 for i,r in enumerate(gdf_state.region)}
    gdf_census["r_idx"] = gdf_census["region"].map(census_rgn_idx)
    df_infz_census["r_idx"] = df_infz_census["region"].map(census_rgn_idx)
    gdf_state["r_idx"] = gdf_state["region"].map(state_rgn_idx)
    df_infz_state["r_idx"] = df_infz_state["region"].map(state_rgn_idx)

    # Rearrage dataframe accorging to date and r_idx
    df_infz_census.sort_values(by = ["date","r_idx"],inplace = True)
    df_infz_state.sort_values(by = ["date","r_idx"], inplace = True)

    # Remove dates before 2020 - Covid period has issues peridicting
    cut_off_date = pd.to_datetime("2020-12-31")
    df_infz_census = df_infz_census[df_infz_census.date <= cut_off_date]
    df_infz_state = df_infz_state[df_infz_state.date <= cut_off_date]
    print(f"df shapes, census : {df_infz_census.shape} , state : {df_infz_state.shape}")

    # Ensure all regions are consistent 
    assert len(np.setdiff1d(df_infz_census.region.unique(), gdf_census.region)) == 0
    assert len(np.setdiff1d(df_infz_state.region.unique(), gdf_state.region)) == 0
    assert len(np.setdiff1d(gdf_census.region, df_infz_census.region.unique())) == 0
    assert len(np.setdiff1d(gdf_state.region, df_infz_state.region.unique())) == 0

    # ==============================================================================
    #* For spatial modelling we need to remove seasonality - Just consider the summer or winter period

    #todo We are removing seasonality by just taking the peak but we need to incorporate it
    #todo PEHAPS we need to use the seasonal pattern from census to respective states
    # Look at histogram here for infz seasons : https://www.cdc.gov/flu/about/season.html
    infz_season = ["December","January","February","March"] 
    df_infz_census["monthname"] = df_infz_census["date"].dt.month_name()
    df_infz_state["monthname"] = df_infz_state["date"].dt.month_name()
    df_infz_census = df_infz_census[df_infz_census.monthname.isin(infz_season)]
    df_infz_state = df_infz_state[df_infz_state.monthname.isin(infz_season)]

    df_clim["monthname"] = df_clim["date"].dt.month_name()
    df_clim_winter = df_clim[df_clim.monthname.isin(infz_season)]
    df_clim_winter = df_clim_winter.groupby(["geoid","county_name"]).agg({
        "sph" : "mean",
        "tmmx_C" : "mean",
        "tmmn_C" : "mean"
    }) #(3060,3)
    df_clim_winter.reset_index(inplace = True)
    df_clim_winter.rename(columns = {"county_name" : "county"}, inplace = True)
    df_clim_winter["geoid"] = df_clim_winter["geoid"].astype(str)
    # join with gdf_county as their a couple of counties where 2 or more regions map to one county
    # Before join ensure geoid matches between both dataframe
    np.setdiff1d(df_clim_winter.geoid.unique(), gdf_county.geoid.unique())
    np.setdiff1d(gdf_county.geoid.unique(), df_clim_winter.geoid.unique())
    np.setdiff1d(df_clim_winter.county.unique(), gdf_county.county.unique())
    np.setdiff1d(gdf_county.county.unique(), df_clim_winter.county.unique())
    # Handle 'Doña Ana' unicode character
    df_clim_winter["county"] = df_clim_winter["county"].replace("Doña Ana", "Dona Ana")
    gdf_county_clim = pd.merge(gdf_county, df_clim_winter, how = "left", on = ["geoid","county"])
    assert gdf_county_clim.shape[0] == 3077
    # ==============================================================================


    plot_region(df_infz_census, 0, "total_cases")
    plot_region(df_infz_state, 9, "total_cases")
    gdf_county_clim.plot(column = "tmmn_C")
    gdf_county_clim.plot(column = "sph")

    # County level grid 
    coords = gdf_county.filter(items = ["centroid_x", "centroid_y"])
    pol_pts_lo, pt_which_pol_lo = compute_pts_in_polys(coords, gdf_census)
    pol_pts_hi, pt_which_pol_hi = compute_pts_in_polys(coords, gdf_state)

    gdf_county["census_idx"]  = pt_which_pol_lo
    gdf_county["state_idx"] = pt_which_pol_hi + 9

    model_path = "model_outputs/v5/gp_vae_2lyr_h1538_h769_z384_ep20_bta0.8/dec_wts"
    with open(model_path, "rb") as f:
        dec_params = pickle.load(f)

    # ==============================================================================
    # Prevalence model (just spatial)
    #* Prevalence field calculated at county level and then aggregate
    #* Climate data at county level when contributing to the linear predictor
    # ==============================================================================
    # Covariates
    x_sph = jnp.array(gdf_county_clim.sph) #(3077, )
    x_tmmn = jnp.array(gdf_county_clim.tmmn_C) #(3077,)
    x_tmmn_std = (x_tmmn - x_tmmn.mean()) / x_tmmn.std() #(3077,)
    x_sph_std = (x_sph - x_sph.mean()) / x_sph.std() #(3077,)
    # Region index vars
    lo_rgn_idx = jnp.array(df_infz_census.r_idx) #(801,)
    hi_rgn_idx = jnp.array(df_infz_state.r_idx) #(3946,)
    # Number of tests and positive cases
    n_tst_lo = jnp.array(df_infz_census.total_specimens.values) #(801,)
    n_tst_hi = jnp.array(df_infz_state.total_specimens.values) #(3946,)
    n_pos_lo = jnp.array(df_infz_census.total_cases.values) #(801,)

    pred_key, sample_key, split_key = random.split(random.PRNGKey(0),3)
    predictive = Predictive(prev_vae_betabin, num_samples=100)

    predictive(
        pred_key, 
        dec_params = dec_params, 
        x_tmmn = x_tmmn, 
        x_sph = x_sph,
        M_lo = pol_pts_lo,
        M_hi = pol_pts_hi,
        lo_rgn_idx = lo_rgn_idx,
        hi_rgn_idx = hi_rgn_idx,
        n_tst_lo = n_tst_lo,
        n_tst_hi = n_tst_hi,
        n_pos_lo = n_pos_lo
    )

    mcmc = MCMC(NUTS(prev_vae_sp_clim_census_state), num_warmup=1000, num_samples = 10000)
    mcmc.run(
        sample_key, 
        dec_params = dec_params, 
        x_tmmn = x_tmmn_std, 
        x_sph = x_sph_std,
        M_lo = pol_pts_lo,
        M_hi = pol_pts_hi,
        lo_rgn_idx = lo_rgn_idx,
        hi_rgn_idx = hi_rgn_idx,
        n_tst_lo = n_tst_lo,
        n_tst_hi = n_tst_hi,
        n_pos_lo = n_pos_lo
    )

    pos_samples = mcmc.get_samples()

    plot_true_est_prev(df_infz_census, pos_samples)
    plot_true_est_prev(df_infz_state, pos_samples)
