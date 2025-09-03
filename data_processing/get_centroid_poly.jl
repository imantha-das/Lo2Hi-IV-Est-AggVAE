import Pkg 
Pkg.activate("jl_env/lo2hi")
using DataFrames
import GeoDataFrames
import LibGEOS
import Shapefile
using CairoMakie
import GeoMakie
import GeometryOps

# Load Shape file for further processing
shp_root = "data/interim/county_edited"
shp_file = filter(x -> endswith(x,".shp"), readdir(shp_root))
shp_p = joinpath(shp_root, first(shp_file))
gdf = GeoDataFrames.read(shp_p)

# Compute centroid of each polygon, each point will be used to evaluate a gp
centroid_x = Float64[]
centroid_y = Float64[]

c = LibGEOS.centroid(gdf.geometry[1])
LibGEOS.getcoord(c,1) |> typeof

for geom in gdf.geometry
    c = LibGEOS.centroid(geom)
    push!(centroid_x, LibGEOS.getcoord(c,1))
    push!(centroid_y,LibGEOS.getcoord(c,2))
end

gdf[!, "centroid_x"] = centroid_x 
gdf[!, "centroid_y"] = centroid_y

# Plot US mainland and centroid of each location 
begin
    f = Figure(size = (1200, 800))
    us_centroid = (-98.5795, 39.8283)
    us_centroid = GeometryOps.centroid(gdf.geometry)
    ax = GeoMakie.GeoAxis(
        f[1,1], 
        dest = "+proj=ortho +lon_0=$(us_centroid[1]) +lat_0=$(us_centroid[2])",
        title = "GP evaluting points at County centroids"
    )
    GeoMakie.poly!(
        ax, gdf.geometry,
        color = (:lightblue, 0.5)
    )
    scatter!(ax, 
        gdf.centroid_x, gdf.centroid_y,
        markersize = 3, color = (:red)
    )
    ax.xticks = Makie.WilkinsonTicks(5; k_min = 4)
    ax.yticks = Makie.WilkinsonTicks(10; k_min = 6)
    f
end

# Rename columns for easier access 
rename!(
    gdf,
    "STATEFP" => :statefp,
    "COUNTYFP" => :countyfp,
    "GEOID" => :geoid, 
    "NAME" => :name,
    "ALAND" => :area_land,
    "AWATER" => :area_water
)

# Select required columns
select!(gdf, :statefp, :countyfp, :geoid, :name, :area_land, :area_water, :centroid_x, :centroid_y, :geometry)

# Save processed geodataframe as a shapefile 
save_root = "data/processed/county_grid"
if !isdir(save_root)
    mkpath(save_root)
end 

GeoDataFrames.write(joinpath(save_root, "county_grid.shp"), gdf)
