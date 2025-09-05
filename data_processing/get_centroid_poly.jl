import Pkg 
Pkg.activate("jl_env/lo2hi")
using DataFrames
import GeoDataFrames
import LibGEOS
import Shapefile
using CairoMakie
import GeoMakie
import GeometryOps
using ArchGDAL: IGeometry

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

# -------------------------- Compute Polygon Points -------------------------- #
# We are doing the inverse, given a grid point can we find which polygon 
# This is useful when we look at larger state areas, as many counties makeup a state or census level
coords = [(row.centroid_x, row.centroid_y) for row in eachrow(gdf)]
poly_regions = gdf.geometry

@doc """
Computes which points fall into which polygons 
Inputs 
    - coords : Lat/Lon value of points 
    - poly_regions : Regions expressed as geometry object
Outputs
    - pol_pts : A 
"""->
function compute_pts_polygons(coords::Vector{Tuple{Float64,Float64}},poly_regions::Vector{IGeometry}) 
    n_pol = length(poly_regions)
    n_pts = length(coords)
    pol_pts = zeros(Int, (n_pol, n_pts))
    pt_which_pol = zeros(Int, n_pts)

    # Loop through polygons i.e 1..9
    for i_pol = 1:n_pol
        pol = poly_regions[i_pol]
        for j_pt = 1:n_pts
            pt = coords[j_pt]
            if LibGEOS.contains(pol,pt)
                pol_pts[i_pol, j_pt] = 1
                pt_which_pol[j_pt] = i_pol
            end
        end
    end
    return pol_pts, pt_which_pol
end

pol_pts, pt_which_pol = compute_pts_polygons(coords, poly_regions)

n_pol = nrow(gdf) #3107 grid points 
n_pts = length(coords) #3107 grid point
pl_pt = zeros(Int,(n_pol, n_pts))
pt_which_pol = zeros(Int,n_pts)


for i_pol in 1:n_pol
    pol = poly_regions[i_pol]
    for j_pt in 1:n_pts 
        pt = coords[j_pt]
        if LibGEOS.contains(pol,pt)
            pl_pt[i_pol, j_pt] = 1 # matrix just say if pt is contained within region, note the is [n_regions, n_points]
            pt_which_pol[j_pt] = i_pol
        end
    end 
end




    

