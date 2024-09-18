# ==============================================================================
# Following scripts prepare data - shape files and case data for aggVAE
# We will remove "Pacific Region" which unfortiunately includes Califonia, Oregan
# and Washington regions. This is because shape files include all these regions
# as under one area called "Pacific"
# ==============================================================================

# ---------------------------------- imports --------------------------------- #
import os 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------- #
#                              Low Resolution Data                             #
# ---------------------------------------------------------------------------- #
# ------------------------------- Helper Funcs ------------------------------- #
def remove_row(df: pd.DataFrame, col_name:str, row_value:str)->pd.DataFrame:
    return df.drop(df[df[col_name] == row_value].index, axis = 0)

def write_to_folder(df:gpd.GeoDataFrame, fold_p:str, file_n:str)-> None:
    if not os.path.exists(fold_p):
        os.mkdir(fold_p)
    df.to_file(os.path.join(fold_p,file_n))

# ------------------ Process Low Resolution Shape File Data ------------------ #
# Read Data
shp_lo_p = "data/interim/low/us_continental_census_division_2018/census_division_2018.shp"
df_shp_lo = gpd.read_file(shp_lo_p)

# Rename and Select only useful columns
df_shp_lo.rename({"NAME" : "area"}, axis = 1,  inplace = True)
df_shp_lo = df_shp_lo.filter(["area", "geometry"])

# Rename area values as fullnames of regions havent been captured properly 
df_shp_lo["area"] =  df_shp_lo["area"].replace("East North", "East North Central")
df_shp_lo["area"] =  df_shp_lo["area"].replace("East South", "East South Central")
df_shp_lo["area"] =  df_shp_lo["area"].replace("Middle Atl", "Middle Atlantic")
df_shp_lo["area"] =  df_shp_lo["area"].replace("New Englan", "New England")
df_shp_lo["area"] =  df_shp_lo["area"].replace("South Atla", "South Atlantic")
df_shp_lo["area"] =  df_shp_lo["area"].replace("West North", "West North Central")
df_shp_lo["area"] =  df_shp_lo["area"].replace("West South", "West South Central")


# ------------------------------ Process Flu Low ----------------------------- #
flu_lo_p = "data/raw/flu_net/census_22to23/WHO_NREVSS_Public_Health_Labs.csv"
df_flu_lo = pd.read_csv(flu_lo_p, skiprows = 1)


# Rename
df_flu_lo.rename(columns = {
    "REGION" : "area", 
    "TOTAL SPECIMENS" : "total_specimens",
    "A (2009 H1N1)" : "A_H1N1",
    "A (H3)" : "A_H3",
    "A (Subtyping not Performed)" : "A",
    "A (H5)" : "A_H5"
}, inplace = True)
# Aggregate flu numbers
cols_to_agg = ["total_specimens", "A_H1N1", "A_H3", "A", "B", "BVic", "BYam", "H3N2v", "A_H5"]
df_flu_lo = df_flu_lo.groupby("area")[cols_to_agg].sum()
df_flu_lo.reset_index(inplace = True)
# Rename "Mid-Atlantic" to "Middle Atlantic"   
df_flu_lo["area"] = df_flu_lo["area"].replace("Mid-Atlantic", "Middle Atlantic")
df_flu_lo["tot_cases"] = df_flu_lo.apply(
    lambda row: row.A_H1N1 + row.A_H3 + row.A + row.B + row.BVic + row.BYam + row.H3N2v + row.A_H5,
    axis = 1
)
df_flu_lo = df_flu_lo.filter(["area", "total_specimens", "tot_cases"])
# Geopandas doesnt allow to save columns names > 10 chars
df_flu_lo.rename(columns = {"total_specimens" : "tot_specs"}, inplace = True)
    
# --------------------------- Combine Low Admin Data --------------------------- #
df_low = df_shp_lo.merge(df_flu_lo, how = "left", left_on="area", right_on = "area")
print(df_low)
print(df_flu_lo)
    
# ------------------------------ Write to folder ----------------------------- #
fold_p = "data/processed/low"
file_n = "us_census_divisions" 
#write_to_folder(df_low, fold_p=fold_p, file_n=file_n)

# ==========================================================================
# High Resolution Data
# ==========================================================================

# ----------------------------- Process Shape High ---------------------------- #
# Read Data
shp_hi_p = "data/raw/shp_files/cb_2018_us_state_5m/cb_2018_us_state_5m.shp"
df_shp_hi = gpd.read_file(shp_hi_p)

# Remove Rows
# rows_to_remove = ["Alaska", "California", "Oregon", "Washington", "Hawaii", "American Samoa", "Puerto Rico",
#                     "United States Virgin Islands", "Guam", "Commonwealth of the Northern Mariana Islands"]
rows_to_remove = ["Alaska", "Hawaii", "American Samoa", "Puerto Rico",
                    "United States Virgin Islands", "Guam", "Commonwealth of the Northern Mariana Islands"]

for row_name in rows_to_remove:
    df_shp_hi = remove_row(df_shp_hi, col_name="NAME", row_value=row_name)
df_shp_hi.reset_index(inplace = True)

# Rename Columns
df_shp_hi.rename(columns = {"NAME" : "area"}, inplace = True)
# Filter required columns
df_shp_hi = df_shp_hi.filter(["area","geometry"])

# ----------------------------- Process Flu High ----------------------------- #
# Read data
flu_hi_p = "data/raw/flu_net/state_22to23/WHO_NREVSS_Public_Health_Labs.csv"
df_flu_hi = pd.read_csv(flu_hi_p, skiprows=1)

# Remove Rows
# rows_to_remove = ["Alaska", "California", "Oregon", "Washington", "Hawaii", 
#                   "American Samoa", "Puerto Rico", "Virgin Islands", 
#                   "New York City"]  
rows_to_remove = ["Alaska", "Hawaii","American Samoa", "Puerto Rico", "Virgin Islands", 
                    "New York City"]

for row_name in rows_to_remove:
    df_flu_hi = remove_row(df_flu_hi, col_name="REGION", row_value=row_name)

# Rename
df_flu_hi.rename(columns = {
    "REGION" : "area", 
    "TOTAL SPECIMENS" : "total_specimens",
    "A (2009 H1N1)" : "A_H1N1",
    "A (H3)" : "A_H3",
    "A (Subtyping not Performed)" : "A",
    "A (H5)" : "A_H5"
}, inplace = True)

# Columns in State wise are already aggregated
for col_name in ["total_specimens", "A_H1N1", "A_H3", "A", "B", "BVic", "BYam", "H3N2v", "A_H5"]:
    df_flu_hi[col_name] = df_flu_hi[col_name].astype("int")

# Compute total flu cases
df_flu_hi[[]]
df_flu_hi["total_flu_cases"] = df_flu_hi.apply(
    lambda row: row.A_H1N1 + row.A_H3 + row.A + row.B + row.BVic + row.BYam + row.H3N2v + row.A_H5,
    axis = 1
)

# Select Required Columns
df_flu_hi = df_flu_hi.filter(["area", "total_specimens", "total_flu_cases"])
# Rename Columns as GeoPandas doesnt save > 10chars
df_flu_hi.rename(columns = {"total_specimens" : "tot_specs", "total_flu_cases" : "tot_cases"}, inplace = True)

    # --------------------------- Combine High Admin Data --------------------------- #
df_hi = df_shp_hi.merge(df_flu_hi, how = "left", left_on="area", right_on = "area")
df_hi.reset_index(inplace = True)
print(df_hi)

# # ------------------------------ Write to folder ----------------------------- #
fold_p = "data/processed/high"
file_n = "us_state_divisions"

write_to_folder(df_hi, fold_p=fold_p, file_n=file_n)
    