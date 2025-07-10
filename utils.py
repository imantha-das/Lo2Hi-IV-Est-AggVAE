from pathlib import Path, PosixPath
import numpy as np 
import geopandas as gpd
from jaxtyping import Array, Float, Integer, Bool
import jax.numpy as jnp
from typing import Dict
import plotly.express as px

# ---------------------------- Computational Grid ---------------------------- #
def get_comp_grid(path:PosixPath)->Dict[str,np.ndarray]:
    """
    Gets computational which is saved at a prior stage

    - x : latitude longitude coordinates
        - shape :(num_grid_pts, lat/lon) i.e (2618,2)

   - pol_pt_xx : Informs weather a point is in the region (0 or 1). For
   example values that take 1 in the following dimension (5,:)
   refers to all points inside region 5
        - shape : (num_regions, num_grid_pts) i.e (9,2618) for low or (49,2618) for hi

    - pt_which_pol_xx : Will take values from 0::num_regios
    For example every point that fall on region 9 will consist
    of a value 9.
        - shape : (num_grid_points,)

    suffix "xx" corresponds to lo or hi, lo consist of 9 regions
    and hi 49 regions.
    """ 
    # Lat/Lon values of the artficial grid
    x = np.load(path / "lat_lon_x.npy") #(num_grid,2) i.e (2618,2)
    # Coarse/Low resolution administrative region
    # Points that fall on region takes a value 1 else 0
    pol_pt_lo = np.load(path / "low" / "pol_pt_lo.npy") #(num_regions_low, grid_size)
    # Points that fall on region take values 0::num_regions
    pt_which_pol_lo = np.load(path / "low" / "pt_which_pol_lo.npy")
    # High resolution administrative region
    pol_pt_hi = np.load(path / "high" / "pol_pt_hi.npy")
    pt_which_pol_hi = np.load(path / "high" / "pt_which_pol_hi.npy")
    # Geopandas dataframes
    df_lo = gpd.read_file(path / "low" / "us_census_divisions" / "us_census_divisions.shp")
    df_hi = gpd.read_file(path / "high" / "us_state_divisions" / "us_state_divisions.shp")
    
    return {
        "x" : x,
        "pol_pt_lo" : pol_pt_lo,
        "pt_which_pol_lo" : pt_which_pol_lo,
        "pol_pt_hi" : pol_pt_hi,
        "pt_which_pol_hi" : pt_which_pol_hi,
        "df_lo" : df_lo,
        "df_hi" : df_hi
    }

# ---------------------------- Kernel And Utility ---------------------------- #
def dist_euclid(x:np.ndarray,z:np.ndarray)->Array:
    """
    Computes the eucledian distance between regions, this funvtion
    is used to find the distance between points.
    Inputs
        - x : lat/lon values
        - z : lat/lon values
    Note for this problem x and z are the same as we are trying to find
    the distance between each point and all other points.
    """ 
    # Convert x,z to jax Arrays
    x:Float[Array, "n_grid_pts 2"] = jnp.array(x)
    z:Float[Array, "n_grid_pts 2"] = jnp.array(z)
    if len(x.shape) == 1:
        x = x.reshape(x.shape[0], 1) #if (num_grid_pts,) -> (num_grid_pts, 1)
    if len(x.shape) == 1:
        z = z.reshape(z.shape[0], 1) # if (num_grid_pts,) -> (num_grid_pts, 1)
    # Get dimensions of x and z
    n_x, m = x.shape # num_grid_pts , 2
    n_z, m_z = z.shape # num_grid_pts , 2
    assert m == m_z 
    # construct a matrix with shape (num_grid_pts,num_grid_pts)
    delta:Float[Array, "n_grid_pts n_grid_pts"] = jnp.zeros((n_x,n_z))
    for d in jnp.arange(m):
        # In 1st iter takes grid_pts_lon in 2nd iter grid_pts_lat
        x_d:Float[Array, "n_grid_pts"] = x[:,d]
        z_d:Float[Array, "n_grid_pts"] = x[:,d]
        # Compute the distance between each point and all points
        delta += (x_d[:,jnp.newaxis] - z_d)**2
        
    return jnp.sqrt(delta) #(num_grid_pts, num_grid_pts)

def exp_sq_kernel(x,z,var,length,noise,jitter =1.0e-4):
    """
    Exponential Square Kernel
    - x : Lat
    - z : Grid 
    """
    dist:Float[Array, "n_grid_pts n_grid_pts"] = dist_euclid(x,z) #e.g (2618,2618)
    deltaXsq:Float[Array, "n_grid_pts n_grid_pts"] = jnp.power(dist / length, 2.0) # e.g (2618,2618)
    k:Float[Array, "n_grid_pts n_grid_pts"] = var * jnp.exp(-0.5 * deltaXsq) # e.g (2618,2618)
    k += (noise + jitter) * jnp.eye(x.shape[0]) # e.g (2618,2618)
    return k

# -------------------------- Aggregate Regions Func -------------------------- #
def M_g(M:Float[Array, "n_regions n_grid_pts"],g:Float[Array, "n_grid_pts"]):
    """ 
    Used to aggregated values per resion
    - M : Matrix with binary entries $m_{ij}, $ showing whether point $j$ is in polygon $i$
    - g : Is a vector of GP draws over the grid
    - matmul(M,g) gives a vector sum over each polygon
    """
    M = jnp.array(M) #e.g (9, 2618) for lo
    g = jnp.array(g).T #e.g (2618,) 
    Mg:Float[Array, "n_regions n_samples"] = jnp.matmul(M,g) #e.g (9,) for lo
    return Mg

# --------------------------- Plot aggregated GP's --------------------------- #

def plot_agg_gps(gp:Float[Array, "n_samples n_regions"]):
    n_samples, n_grids = gp.shape 
    p = px.line()
    for i in range(n_samples):
        p.add_scatter(
            x = jnp.arange(n_grids), y = gp[i,:], 
            mode = "lines", 
            line = dict(color = "black"),
            opacity = 0.4
        )
    p.update_layout(
        template = "plotly_white", 
        title = "GP samples", 
        xaxis_title = "regions", yaxis_title = "spatial gp",
        showlegend = False
    )

    p.show()