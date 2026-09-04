import jax 
import jax.numpy as jnp 
import jax.random as random 
from jaxtyping import Float, Array 

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
    - x : Lat lon as 2D vector [n_pts, 2]
    - z : lat lon as 2D vector [n_pts, 2] (this is the same x here)
    """
    dist:Float[Array, "n_grid_pts n_grid_pts"] = euclid_dist(x,z) #e.g (2618,2618)
    deltaXsq:Float[Array, "n_grid_pts n_grid_pts"] = jnp.power(dist / length, 2.0) # e.g (2618,2618)
    k:Float[Array, "n_grid_pts n_grid_pts"] = var * jnp.exp(-0.5 * deltaXsq) # e.g (2618,2618)
    k += (noise + jitter) * jnp.eye(x.shape[0]) # e.g (2618,2618)
    return k

def compute_pts_in_polys(coords,df_reg):
    """
    Computes if a point (county level grid) fall on census/state level polygon.
    Params
        - coords : county centroid as lat/lon values 
        - df_reg : geopandas census or state level regions dataframe
    Out
        - pol_pts : (n_regs, n_pts) jax array containing 1/0 of point falls
        on census / state polygon
        - pt_which_pol : (n_pts,) jax array containing a regions id (0:n_regions)
        States what region each point belongs to
    """
    uniq_regs = df_reg.region.unique()
    df_reg["reg_id"] = range(0, len(uniq_regs)) 

    n_pts,_ = coords.shape
    n_pols = len(uniq_regs)
    pol_pts = jnp.zeros(shape = (n_pols,n_pts))
    pt_which_pol = jnp.zeros(shape = (n_pts,))
    for reg in uniq_regs:
        df_flt = df_reg[df_reg.region == reg]
        reg_id = int(df_flt["reg_id"].values[0])
        geoms = df_flt.geometry 
        for i_pt in range(n_pts):
            pt = Point(coords.iloc[i_pt])
            for pol in geoms:
                if pol.contains(pt):
                    pol_pts = pol_pts.at[reg_id, i_pt].set(1) #(n_regs, n_pts)
                    pt_which_pol = pt_which_pol.at[i_pt].set(reg_id) #(n_pts,)
    return pol_pts, pt_which_pol