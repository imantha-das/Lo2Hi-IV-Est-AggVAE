import os 
import numpy as np 
import geopandas as gpd

import torch 
import torch.nn as nn 

import pyro 
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive

from typing import Dict, List, Any, Union
import plotly.express as px

import time

DEVICE = "cuda" if torch.cuda.is_available else "cpu"

def dist_euclid(x:torch.Tensor,z:torch.Tensor)->torch.Tensor:
    """
    x : Lat/Lon values, Shape : (*,2)
    z : Also lat/lon values in this situation 
    """
    x = torch.tensor(x, dtype = torch.float32).to(DEVICE) #shape : (num_grid_pts, 2)
    z = torch.tensor(z, dtype = torch.float32).to(DEVICE) #shape : (num_grid_pts, 2)
    if len(x.shape) == 1:
        x = x.reshape(x.shape[0], 1) #shape : (num_grid_pts, 1)
    if len(z.shape) == 1:
        z = z.reshape(z.shape[0], 1) #shape : (num_grid_pts, 1)
    n_x , m = x.shape
    n_z , m_z = z.shape 
    assert m == m_z
    delta = torch.zeros((n_x, n_z)).to(DEVICE) #shape : (num_grid_pts, num_grid_pts)
    for d in torch.arange(m): #num_grid_pts ; i.e 0..1919
        x_d = x[:, d] #take col vector from sq mat
        z_d = z[:, d]
        delta = (x_d.unsqueeze(1) - z_d)**2 
    return torch.sqrt(delta) #shape : (num_grid_pts, num_grid_pts)  

def exp_sq_kernel(x:torch.Tensor,z:torch.Tensor,var:float, length:float, noise:float, jitter = 1e-4)->torch.Tensor:
    """
    Kernel Function
    Inputs 
        - x : Lat/Lon values of grid points
        - z : Lat/Lon values of grid points
        - var : variance hyperparam for kernel
        - length : length hyperparam for kernel
        - noise : noise
        - jitter : some noise for numerical stability
    """
    dist = dist_euclid(x,z) #shape : (num_grid_pts, num_grid_pts)
    deltaXsq = torch.pow(dist/length, 2.0) #square dist
    k = var * torch.exp(-0.5 * deltaXsq)
    k += (noise + jitter) * torch.eye(x.shape[0]).to(DEVICE)
    return k #shape : (num_grid_points, num_grid_points)

def M_g(M:Union[torch.Tensor, np.ndarray],g:Union[torch.Tensor, np.ndarray]):
    """
    Aggregation function
    Inputs
        - M : Matrix with binary entries m_{ij} show wheather point j is in polygon i 
        - g : Vector of Gp draws over grid 
    Output
        - matmul(M,g) : Vector of sums over each polygon
    """
    if not isinstance(M, torch.Tensor):
        M = torch.tensor(M, dtype = torch.float32).to(DEVICE) #shape : (num_regions, num_grid_pts)
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g, dtype = torch.float32).to(DEVICE) #shape : (num_grid_pts,)
    
    return torch.matmul(M,g) #shape : (num_regions,)

#todo We should pass n_obs for y , we will have to this manually by changing last line pyro.sample(..., obs = n_obs)   
def prev_model_gp_aggr(args:Dict[str, Any], y:Union[np.ndarray, torch.tensor]=None):
    """
    Aggregated GP
    Inputs
        - args : Dictionary containing the following 
            - n_obs : number of observations/cases ; shp : (num_all_regions,) <- all regions include both hi res and low res regions
            - x : lat/lon grid point values ; shp : (num_grid_pts, 2)
            - gp_kernel : exponential kernel function
            - jitter : random noise for numerical stability ; scaler
            - M1 (M_lo/M_old) : Matrix with binary entries m_{ij} show wheather point j is in polygon i 
            - M2 (M_hi/M_new) : Matrix with binary entries m_{ij} show wheather point j is in polygon i 
    """

    n_obs = args["n_obs"]
    x = args["x"]
    gp_kernel = args["gp_kernel"]
    jitter = args["jitter"]
    noise = args["noise"]
    M1 = args["M1"] #old/low
    M2 = args["M2"] #new/hi

    length = pyro.sample("kernel_length", dist.InverseGamma(3,3))
    var = pyro.sample("kernel_var", dist.HalfNormal(0.05))

    if not isinstance(n_obs, torch.Tensor):
        n_obs = torch.tensor(n_obs, dtype = torch.float32).to(DEVICE)
    if y is not None:
        if not isinstance(y, torch.Tensor):
            print("executed")
            print(y)
            y = torch.tensor(y, dtype = torch.float32).to(DEVICE)

    # GP Kernel
    k = gp_kernel(x,x,var,length,noise,jitter)
    # GP draw
    f = pyro.sample(
        "f", 
        dist.MultivariateNormal(loc = torch.zeros(x.shape[0]).to(DEVICE), covariance_matrix = k)
    )

    # Aggregate
    gp_aggr_m1 = pyro.deterministic("gp_aggr_m1", M_g(M1,f)) #shp : (num_regions_m1,)
    gp_aggr_m2 = pyro.deterministic("gp_aggr_m2", M_g(M2,f)) #shp : (num_regions_m2,)
    gp_aggr = pyro.deterministic("gp_aggr", torch.cat([gp_aggr_m1, gp_aggr_m2])) #shp : (num_regions_m1 + num_regions_m2,)

    # Fixed Edffects
    b0 = pyro.sample("b0", dist.Normal(0,1)) #scaler
    # Linear Predictor
    lp = b0 + gp_aggr #(num_regions_m1 + num_regions_m2 ,)
    theta = pyro.deterministic("theta", torch.sigmoid(lp)) #(num_regions_m1 + num_regions_m2 ,)

    pyro.sample("obs", dist.Binomial(total_count = n_obs, logits = lp),  obs = n_obs) #shp :(1, num_grid_pts)


if __name__ == "__main__":
    # ==========================================================================
    # Load Data
    # ==========================================================================
    data_dir = "misc/interim_saved_data_var"
    
    # Lat/Lon Values of artificial grid
    x = np.load(os.path.join(data_dir,"x.npy"))
    pol_pt_new = np.load(os.path.join(data_dir, "pol_pt_new.npy"))
    pol_pt_old = np.load(os.path.join(data_dir, "pol_pt_old.npy"))
    s_new = gpd.read_file(os.path.join(data_dir, "s_new", "s_new.shp"))
    s_old = gpd.read_file(os.path.join(data_dir, "s_old", "s_old.shp"))
    s = gpd.read_file(os.path.join(data_dir, "old_new_geom_cases","old_new_geom_cases.shp"))

    # ==========================================================================
    # Test Section
    # ==========================================================================
    var = dist.HalfNormal(0.05).sample().item()
    length = dist.InverseGamma(3,3).sample().item()
    noise = 1e-4
    #print(var, length, noise)
    k = exp_sq_kernel(x,x,var, length, noise)
    p1 = px.imshow(k.to("cpu"))
    #p1.show()

    # sample a gaussian function
    f = dist.MultivariateNormal(loc = torch.zeros(x.shape[0]).to(DEVICE), covariance_matrix = k).sample()
    # Test M_g func
    
    mg = M_g(pol_pt_new, f)

    # ==========================================================================
    # Test Prev Model Aggr
    # ==========================================================================
    args = {
        "n_obs" : s.n_obs, # shape : (116,) ; number of cases
        "x" : x, # shape : (1920,2)
        "z" : x, # shape : (1920,2)
        "gp_kernel" : exp_sq_kernel,
        "batch_size"  : 1,
        "jitter" : 1e-4,
        "noise" : 1e-4,
        "M1" : pol_pt_old, # shape : (69, 1920) ; old
        "M2" : pol_pt_new # shape : (47,1920) ; new
    }

    p2 = px.line()
    for i in range(10):
        prev_predictive = Predictive(prev_model_gp_aggr, num_samples = args["batch_size"])
        gp_draw = prev_predictive(args)["f"]
        p2.add_scatter(x = np.arange(0, len(gp_draw.ravel())), y = gp_draw.ravel().to("cpu"), line_color = 'black', line_width = 1)
    p2.update_layout(template = "plotly_white", xaxis_title = "grid_point", yaxis_title = "number of case/obs", title = "GP Draws")
    #p2.show()

    # ==========================================================================
    # MCMC
    # ==========================================================================
    kernel = NUTS(model = prev_model_gp_aggr)
    n_warm = 200
    n_samps = 1000
    mcmc_gp_aggr = MCMC(kernel, warmup_steps = n_warm, num_samples = n_samps)

    run_mcmc = True 
    if run_mcmc:
        start = time.time()
        mcmc_gp_aggr.run(args, y = torch.tensor(s.y, dtype = torch.float32).to(DEVICE))
        t_elpased = time.time() - start

    if run_mcmc:
        prev_samples_aggr = mcmc_gp_aggr.get_samples()
        mcmc_gp_aggr.summary(exclude_deterministic = False)

        print("\nMCMC elapsed time :", round(t_elpased), "s")
        print("\n MCMC elapsed time :", round (t_elpased/60), "min")
        print("\nMCMC elapsed time:", round(t_elpased/(60*60)), "h")

        ss = numpyro.diagnostics.summary(mcmc_gp_aggr.get_samples(group_by_chain = True))
        r = np.mean(ss["gp_aggr"]["n_eff"])
        print("Average ESS for all aggGP effects : " + str(round(r)))

        r = np.mean(ss["gp_aggr_old"]["n_eff"])
        print("Average ESS for all aggGP-old effects : " + str(round(r)))
        print("Max r_hat for all aggGP-old effects : " + str(round(np.max(ss["gp_aggr_old"]["r_hat"]),2)))

        r = np.mean(ss["gp_aggr_new"]["n_eff"])
        print("Average ESS for all aggGP-new effects : " + str(round(r)))
        print("Max r_hat for all aggGP-ew effects : " + str(round(np.max(ss["gp_aggr_new"]["r_hat"]),2)))

        print("kernel+length R-hat : " + str(round(ss["kernel_length"]["r_hat"],2)))
        print("kernel_var R-hat : " + str(round(ss["kernel_var"]["r_hat"],2)))



    