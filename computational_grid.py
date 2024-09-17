import numpy as np
import pandas as pd
import geopandas as gpd 
import matplotlib.pyplot as plt 

from typing import Tuple, List
import math

def compute_grid(num_grid_x:Tuple[int],extents:Tuple[float]):
    """
    Computes Grid Points 
    num_grid_x : number of grid points in x direction
    extents : Map Extents
    """
    x_min, x_max, y_min, y_max = extents 
    
    dy = y_max - y_min 
    dx = x_max - x_min 
    factor = dy / dx 

    n_x = num_grid_x 
    n_y = math.ceil(factor * num_grid_x)

    x_grid = np.linspace(x_min, x_max, n_x, endpoint=True) #(n_x,)
    y_grid = np.linspace(y_min, y_max, n_y, endpoint=True) #(n_y,)

    # full coordinate arrays
    x_coords, y_coords = np.meshgrid(x_grid, y_grid) #(n_x,n_y),(n_x,n_y) <- Values in these arrays arnt the same
    x_coords = x_coords.reshape(-1) #(n_x * n_y,)
    y_coords = y_coords.reshape(-1) #(n_x * n_y,)

    df = pd.DataFrame({"Latitude" : y_coords, "Longitude" : x_coords})
    grid_pts = gpd.GeoDataFrame(df, geometry= gpd.points_from_xy(df.Longitude, df.Latitude))

    x = np.array([x_coords, y_coords]).transpose((1,0)) #(n_x * n_y , 2)

    return x, grid_pts

def pol_pts(df_shp:gpd.GeoDataFrame, grid_pts:gpd.GeoDataFrame):
    """
    Counts the grid points that fall on the regions
     - df_shp : takes the dataframe containing regions
     - grid_pts : takes the dataframe containing grid_points

    """
    grid_pts.set_crs(epsg=4326, inplace = True) #geometry column remains the same, but originally it doesnt have crs until you set it
    grid_pts.crs == df_shp.crs # ensures both are epsh 4326
    n_pol = len(df_shp.geometry) #46 for s_new
    n_pts = len(grid_pts.geometry) #1919 - num grid points

    pl_pt = np.zeros((n_pol, n_pts), dtype = int) #zeros(46, 1919) for new_regions
    pt_which_pol = np.zeros(n_pts, dtype = int) #(1919,) for s_new possibly the same for s_old 

    for i_pol in range(n_pol): #1..46
        pol = df_shp.geometry[i_pol] #i.e POLYGON((...))
        for j_pts in range(n_pts): # 0 ...1919 
            pt = grid_pts.geometry[j_pts] #i.e point((...))
            # Check if point falls on polygon
            if pol.contains(pt):
                pl_pt[i_pol, j_pts] = 1 # matrix just says if there is point
                # in the code cell below we use this to make a new column that will helps us
                # figure out which region each point corresponds to
                pt_which_pol[j_pts] = i_pol + 1 

    return([pl_pt, pt_which_pol])

def get_points_in_region(df:gpd.GeoDataFrame, grid_pts:gpd.GeoDataFrame):
    """Returns an array with points that fall regions as well as which point that falls
    on region.
    """
    pol_pt, pt_which_pol = pol_pts(df, grid_pts) #(9, num_points), (num_points,)
    pol_sums = np.sum(pol_pt, axis = 1)
    assert all(item > 0 for item in pol_sums), "Region(s) with no pints exists !"
    print("Atleast one point falls on every region !")        
    return pol_pt, pt_which_pol

def check_for_min_points(start_value, end_value, df, extents):
    """Helper to look for a given configuration wher eatleast one point falls
    on every region."""
    num_grid_x = start_value
    for i in range(0,end_value):
        x, grid_pts = compute_grid(num_grid_x, extents)
        pol_pt_hi, pt_which_pol_hi = pol_pts(df, grid_pts) #(9, num_points), (num_points,)
        pol_sums_hi = np.sum(pol_pt_hi, axis = 1)
        if all(item > 0 for item in pol_sums_hi):
            print(f"Min point fall for these extents : {num_grid_x}")
            return x
            break
        else:
            print(num_grid_x)
            num_grid_x += 1



if __name__ == "__main__":
    # Load Data
    df_lo = gpd.read_file("data/processed/low/us_census_divisions/us_census_divisions.shp")
    df_hi = gpd.read_file("data/processed/high/us_state_divisions/us_state_divisions.shp")

    num_grid_x = 77
    # Manually look at map and decide on a grid
    extents = (-125,-67,24.5,49.5)
    x, grid_pts = compute_grid(num_grid_x, extents)
    print(f"Num Grid Points : {grid_pts.shape}")

    pol_pts_hi, pt_which_pol_hi = get_points_in_region(df_hi, grid_pts)
    pol_pts_lo, pt_which_pol_lo = get_points_in_region(df_lo, grid_pts)

    np.save("data/processed/lat_lon_x", x)
    np.save("data/processed/low/pol_pts_lo",pol_pts_lo)
    np.save("data/processed/low/pt_which_pol_lo",pt_which_pol_lo)
    np.save("data/processed/high/pol_pts_hi",pol_pts_hi)
    np.save("data/processed/high/pt_which_pol_hi",pt_which_pol_hi)