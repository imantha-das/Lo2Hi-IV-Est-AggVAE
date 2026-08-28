import os
import numpy as np
import jax 
import jax.numpy as jnp 
import pandas as pd 
import geopandas as gpd
from shapely import Point 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def plot_map_with_points(gdf_reg, coords, pt_which_pol, title, col_by_reg = False):
    """
    Plots a map with polygons and computational grid points.
    Params
        - gdf_reg : Geodataframe containing census/state level regions
        - coords : County level grid point (centroid_x,centroid_y)
        - pt_which_pol : array of regions ids (n_pts,)
        - title : figure title
        - col_by_reg : boolean to color points
    """
    fig, ax = plt.subplots(1,1,figsize = (12,8))
    gdf_reg.plot(
        ax = ax, 
        color = "lightblue",
        edgecolor = "black",
        linewidth = 1,
        alpha = 0.5
    )
    x = coords["centroid_x"].values 
    y = coords["centroid_y"].values 

    if col_by_reg:
        n_regs = len(np.unique(gdf_reg.region))
        cmap = cm.get_cmap("tab20", n_regs)
        colors = [cmap(int(r) % n_regs) for r in pt_which_pol]
        ax.scatter(x,y, c = colors,s = 4)
    else:
        ax.scatter(x,y, color = "red", s = 4)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.show()
    return fig

if __name__ == "__main__":
    root = "data/processed/v4_clin_labs"
    gdf_census = gpd.read_file(os.path.join(root, "census_9","census_9.shp"))
    gdf_state = gpd.read_file(os.path.join(root,"state_47", "state_47.shp"))
    gdf_county = gpd.read_file(os.path.join(root,"county_3099","county_3099.shp"))

    coords = gdf_county.filter(items = ["centroid_x","centroid_y"])     
    pol_pts_lo, pt_which_pol_lo = compute_pts_in_polys(coords,gdf_census)
    pol_pts_hi, pt_which_pol_hi = compute_pts_in_polys(coords,gdf_state)

    print(pol_pts_lo.shape, pt_which_pol_lo.shape)
    print(pol_pts_hi.shape, pt_which_pol_hi.shape)
    print(jnp.unique(pt_which_pol_lo))
    print(jnp.unique(pt_which_pol_hi))

    plot_map_with_points(
        gdf_census,
        coords,
        pt_which_pol_lo,
        "Census Regions and County level grid points",
        col_by_reg= True
    )
    plot_map_with_points(
        gdf_state,
        coords,
        pt_which_pol_hi,
        "State Regions and County level grid points",
        col_by_reg=True
    )