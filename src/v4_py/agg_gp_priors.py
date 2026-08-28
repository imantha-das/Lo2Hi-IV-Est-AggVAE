import os
import jax 
import jax.numpy as jnp 
import jax.random as random
from jaxtyping import Float, Array
import numpyro 
import numpyro.distributions as dist 

import geopandas as gpd 
import plotly.express as px
import plotly.io as pio 
from construct_spatial_matrix import compute_pts_in_polys

pio.templates.default = "plotly_white"

def euclid_dist(x1:Float[Array, "* 2"], x2:Float[Array,"* 2"]):
  """
  Compute Eucledian Distance
  d = sqrt(
    (x2 - x1)^2 + (y2 - y1)^2
  )
  Inputs :
    - x1 : (*,2)
    - x2 : (*,2)
  """

  x = x1[:,0] - jnp.expand_dims(x2[:,0],axis = 1)
  y = x1[:,1] - jnp.expand_dims(x2[:,1],axis = 1)
  d = jnp.sqrt(x**2 + y **2)
  return d 

def exp_sq_kernel(x,z,var,length,noise,jitter =1.0e-4):
    """
    Exponential Square Kernel
    - x : Lat
    - z : Grid 
    """
    dist:Float[Array, "n_grid_pts n_grid_pts"] = euclid_dist(x,z) #e.g (2618,2618)
    deltaXsq:Float[Array, "n_grid_pts n_grid_pts"] = jnp.power(dist / length, 2.0) # e.g (2618,2618)
    k:Float[Array, "n_grid_pts n_grid_pts"] = var * jnp.exp(-0.5 * deltaXsq) # e.g (2618,2618)
    k += (noise + jitter) * jnp.eye(x.shape[0]) # e.g (2618,2618)
    return k

def M_g(
  M:Float[Array, "n_regions n_grid_pts"],
  g:Float[Array, "n_grid_pts"],
  pop = None,
  glb = True
  ):
    """ 
    Used to aggregated values per resion
    - M : Matrix with binary entries $m_{ij}, $ showing whether point $j$ is in polygon $i$
    - g : Is a vector of GP draws over the grid
    - matmul(M,g) gives a vector sum over each polygon
    - Including pop global normalization
    """
    M = jnp.array(M) #e.g (9, 2618) for lo
    g = jnp.array(g).T #e.g (2618,) 
    if pop is not None:
      pop = jnp.array(pop)
      if glb: # weight gp by the national population
        w = pop / jnp.sum(pop) 
        if g.ndim == 1: # if g doesnt have sample dimension (just a single sample)
          g = g * w 
        else: # if g has a batch dimension
          g = g * w[:, None]
      else: # weight gp by within each region pop (within each region normalisation)
        # regional population total
        regional_pop = M @ pop
        # normalise weights within each region, w_ij = pop_j / sum_k(pop_k * M_ik)
        w = pop[None,:] / regional_pop[:, None] # (n_regions, n_pts)
        # Weighted average per region
        return jnp.sum(M * w * g[None, :], axis = 1) # (n_regions,)
    Mg:Float[Array, "n_regions n_samples"] = jnp.matmul(M,g) #e.g (9,) for lo
    return Mg

if __name__ == "__main__":
    root = "data/processed/v4_clin_labs"
    gdf_census = gpd.read_file(os.path.join(root, "census_9","census_9.shp"))
    gdf_state = gpd.read_file(os.path.join(root,"state_47", "state_47.shp"))
    gdf_county = gpd.read_file(os.path.join(root,"county_3099","county_3099.shp"))

    coords = gdf_county.filter(items = ["centroid_x","centroid_y"])     
    pol_pts_lo, pt_which_pol_lo = compute_pts_in_polys(coords,gdf_census)
    pol_pts_hi, pt_which_pol_hi = compute_pts_in_polys(coords,gdf_state)

    x = jnp.array(coords) #(n_pts,2)
    n_pts,_ = x.shape
    # test euclidian distance
    d = euclid_dist(x,x)
    # test exponential square kerenel
    kl = dist.InverseGamma(3,3).sample(random.PRNGKey(0))
    kv = dist.HalfNormal(1.0).sample(random.PRNGKey(1))
    k = exp_sq_kernel(x,x,kv,kl,1e-4)
    
    # Construct GP (This usually happens inside a function)
    n_samples = 1000
    f = dist.MultivariateNormal(
        loc = jnp.zeros(n_pts),
        covariance_matrix= k
      ).sample(random.PRNGKey(2), sample_shape=(n_samples,))

    p = px.line()
    n_lo,_ = pol_pts_lo.shape
    n_hi,_ = pol_pts_hi.shape

    # Plot aggregated GP's
    for i in range(n_samples):
      f_sample = f[i,:]
      gp_agg_lo = M_g(pol_pts_lo, f_sample,pop = gdf_county.pop2020) #(9,)
      gp_agg_hi = M_g(pol_pts_hi, f_sample,pop = gdf_county.pop2020) #(47,)
      p.add_scatter(
        x = jnp.arange(n_lo), 
        y = gp_agg_lo, 
        line = dict(color = "black"), 
        showlegend = False,
        opacity = 0.3
      )
      p.add_scatter(
        x = jnp.arange(n_lo,n_lo+n_hi), 
        y = gp_agg_hi, 
        line = dict(color = "blue"), 
        showlegend = False,
        opacity = 0.3
      )
      p.update_layout(title = "GP Prior (pop weighted)")
    
    p.show()


