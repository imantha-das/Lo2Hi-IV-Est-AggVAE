import Pkg 
Pkg.activate("jl_env/lo2hi")
using DataFrames 
import GeoDataFrames
import CSV 


# -------------------------------- Shape Files ------------------------------- #
shp_lo_p = "data/interim/census_edited/us_mainland_census_crs4326.shp"
shp_hi_p = "data/interim/state_edited/us_mainland_state_crs4326.shp"
df_shp_lo = GeoDataFrames.read(shp_lo_p)
df_shp_hi = GeoDataFrames.read(shp_hi_p)
rename!(df_shp_lo, :NAME => :region, :ALAND => :area_land, :AWATER => :area_water)
rename!(df_shp_hi, :NAME => :region, :ALAND => :area_land, :AWATER => :area_water)
select!(df_shp_lo, [:region,:area_land,:area_water, :geometry])
select!(df_shp_hi, [:region,:area_land,:area_water, :geometry])

# --------------------------------- Flu Data --------------------------------- #

flu_lo_p = "data/raw/fluview_census_yr10to25/ICL_NREVSS_Public_Health_Labs.csv"
flu_hi_p = "data/raw/fluview_state_yr12to25/ICL_NREVSS_Public_Health_Labs.csv"
df_flu_lo = CSV.read(flu_lo_p, DataFrame; header=2)
df_flu_hi = CSV.read(flu_hi_p, DataFrame; header=2)
new_col_names = Dict(
    "REGION" => "region", 
    "TOTAL SPECIMENS" => "test_cases", 
    "A (2009 H1N1)" => "a_2009_h1n1", 
    "A (H3)" => "a_h3", 
    "A (Subtyping not Performed)" => "a_no_subtype", 
    "B" => "b",
    "BVic" => "b_vic", 
    "BYam" => "byam", 
    "H3N2v" => "h3n2v", 
    "A (H5)" => "a_h5" 
)

# --------------------------- Renaming column names -------------------------- #
rename!(df_flu_lo, merge(new_col_names, Dict("YEAR" => "year")))
rename!(df_flu_hi, merge(new_col_names, Dict("SEASON_DESCRIPTION" => "season_description")))
select!(df_flu_lo, Not("REGION TYPE", "WEEK"))
select!(df_flu_hi, Not("REGION TYPE"))

# ------------------ Ensureing Regions Names are Consistent ------------------ #

# Identify different regions between influenza data and shapefiles 
non_ovelap_regions_lo = symdiff(df_shp_lo.region, unique(df_flu_lo.region))
# Mid-Atlantic needs to replaced with Middle Atlantic
replace!(df_flu_lo.region, "Mid-Atlantic" => "Middle Atlantic")

non_overlap_regions_hi = symdiff(df_shp_hi.region, unique(df_flu_hi.region))
filter!(:region => x -> !(x in non_overlap_regions_hi), df_flu_hi)

# Ensure there are non non-overlapping regions 
@assert symdiff(df_shp_lo.region, unique(df_flu_lo.region)) == String[]
@assert symdiff(df_shp_hi.region, unique(df_flu_hi.region)) == String[]

# -------------------------- Handling Missing Values ------------------------- #
any(ismissing, eachcol(df_flu_lo))
for col in [:test_cases, :a_2009_h1n1, :a_h3, :a_no_subtype, :b, :b_vic, :byam, :h3n2v, :a_h5]
    replace!(df_flu_hi[!, col], "x" => missing)
    df_flu_hi[!, col] = parse.(Int, df_flu_hi[!, col])
end

@assert  !any(ismissing, eachcol(df_flu_hi)) 

# --------------------- Get year from season description --------------------- #
transform!(df_flu_hi, 
    :season_description => ByRow(x -> parse(Int, split(x, r"[- ]")[2])) => :year)
select!(df_flu_hi, Not(:season_description))

# ----------------------- Compute total influenza cases ---------------------- #
transform!(df_flu_lo, names(df_flu_lo,Not(:region, :year, :test_cases)) => ByRow(+) => :tested_pos)
transform!(df_flu_hi, names(df_flu_hi,Not(:region, :year, :test_cases)) => ByRow(+) => :tested_pos)

# -------------------- Aggregate Based on region and year -------------------- #
df_flu_lo_aggr = combine(
    groupby(df_flu_lo, [:region, :year]),
    [
        :test_cases => sum => :test_cases,
        :a_2009_h1n1 => sum => :a_2009_h1n1,
        :a_h3 => sum => :a_h3,
        :a_no_subtype => sum => :a_no_subtype,
        :b => sum => :b,
        :b_vic => sum => :b_vic,
        :byam => sum => :byam,
        :h3n2v => sum => :h3n2v,
        :a_h5 => sum => :a_h5,
        :tested_pos => sum => :tested_pos
    ] 
)
df_flu_hi_aggr = combine(
    groupby(df_flu_hi, [:region, :year]),
    [
        :test_cases => sum => :test_cases,
        :a_2009_h1n1 => sum => :a_2009_h1n1,
        :a_h3 => sum => :a_h3,
        :a_no_subtype => sum => :a_no_subtype,
        :b => sum => :b,
        :b_vic => sum => :b_vic,
        :byam => sum => :byam,
        :h3n2v => sum => :h3n2v,
        :a_h5 => sum => :a_h5,
        :tested_pos => sum => :tested_pos
    ] 
)

# Remove year 2025 form low resolution regions as it doesnt exist in high resolution regions 
filter!(:year => !=(2025), df_flu_lo_aggr)
# -------------------------- Save DataFrames as CSV -------------------------- #
save_root = "data/processed/nrevss_phl_20152024"
if !isdir(save_root)
    mkpath(save_root)
end
CSV.write(joinpath(save_root, "infz_census_20152024.csv"), df_flu_lo_aggr)
CSV.write(joinpath(save_root, "infz_state_20152024.csv"), df_flu_hi_aggr)