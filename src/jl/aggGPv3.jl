import Pkg
Pkg.activate("jl_env/lo2hi")
using DataFrames 
import GeoDataFrames 
import LibGEOS
include("utils.jl")
using .Utils: compute_pts_polygons, plot_map_with_points,plot_ax_with_points, M_g
using AbstractGPs: GP, SEKernel, GibbsKernel
using StaticArrays: SVector
using CairoMakie
using GeoMakie:GeoAxis
import GeometryOps
set_theme!(theme_light())


#if abspath(PROGRAM_FILE) == @__FILE__ 
# ------------------------------------ GiS ----------------------------------- #
# Load county : Grid Points will be at this level
county_fold = "data/processed/gis/county_grid"
county_files = filter(x -> endswith(x, ".shp"), readdir(county_fold))
county_path = joinpath(county_fold, first(county_files))
df_county = GeoDataFrames.read(county_path)

# Load Census : Low Resolution data
census_fold = "data/processed/gis/low/us_census_divisions"
census_files = filter(x -> endswith(x,".shp"), readdir(census_fold)) 
census_path = joinpath(census_fold, first(census_files))
df_census = GeoDataFrames.read(census_path)

# Load States : High resolution data 
state_fold = "data/processed/gis/high/us_state_divisions"
state_files = filter(x -> endswith(x, ".shp"), readdir(state_fold))
state_path = joinpath(state_fold, state_files[1])
df_state = GeoDataFrames.read(state_path)

# ---------------------------- Polygons and Points --------------------------- #
coords = zip(df_county.centroid_x, df_county.centroid_y) |> collect
county_polys = df_county.geometry
census_polys = df_census.geometry
state_polys = df_state.geometry
pol_pts_lo, pt_which_pol_lo = compute_pts_polygons(coords, census_polys)
pol_pts_hi, pt_which_pol_hi = compute_pts_polygons(coords, state_polys)

pts = hcat(df_county.centroid_x, df_county.centroid_y) 

fig = plot_map_with_points(
    census_polys, 
    pts, 
    pt_which_pol_lo, 
    "GP evaluating points at census level",
    col_by_reg = false
)

fig = Figure(size = (1200,800), title = "Grid Points at County Level")

us_centroid = GeometryOps.centroid(census_polys)
ax_lo = GeoAxis(
    fig[1,1],
    dest = "+proj=ortho +lon_0=$(us_centroid[1]) +lat_0=$(us_centroid[2])",
    title = "GP evaluation points for Census regions"
)
plot_ax_with_points(ax_lo,census_polys,pts,pt_which_pol_lo,:dodgerblue2)

us_centroid = GeometryOps.centroid(state_polys)
ax_hi = GeoAxis(
    fig[1,2],
    dest = "+proj=ortho +lon_0=$(us_centroid[1]) +lat_0=$(us_centroid[2])",
    title = "GP evaluation points for State regions"
)
plot_ax_with_points(ax_hi,state_polys, pts, pt_which_pol_hi, :darkolivegreen)
fig
save("plots/county_grids.png",fig)

# ---------------------------- Gaussian Processes ---------------------------- #

x = copy(pts) #(3107,2)
k = SEKernel()
f = GP(k)
x_svec = [SVector{2,}(x[i,:]) for i in 1:size(x,1)]
f_latent = f(x_svec, 1e-4)

lo_regions = df_census.area |> collect
hi_regions = df_state.area |> collect
regions = vcat(lo_regions, hi_regions)

gp_aggr_lo = M_g(pol_pts_lo, rand(f_latent, 100)) #(n_reg_lo,n_samps)
gp_aggr_hi = M_g(pol_pts_hi, rand(f_latent, 100)) #(n_reg_hi,n_samps)
gp_aggr = vcat(gp_aggr_lo,gp_aggr_hi) #(n_reg_lo+hi, n_samps)

function plot_gp_aggr(gp_aggr::Matrix{Float64}, regions::Vector{String};n_reg_lo::Int = 9, n_reg_hi::Int = 49, kernel_name::String = "RBF")
    fig = Figure(size = (1200,400), title = "Aggregated Latent GP Realizations (RBF)")
    ax = Axis(fig[1,1])
    for i = 1:size(gp_aggr,2)
        lines!(ax,1:size(gp_aggr,1), gp_aggr[:,i], color = (:black,0.3))
    end
    vspan!(ax, [0],[n_reg_lo], color =(:dodgerblue2, 0.3), label = "low resolution")
    vspan!(ax, [n_reg_lo], [n_reg_lo + n_reg_hi], color = (:darkolivegreen,0.2),label = "high resolution")
    axislegend(positon = :rt)
    ax.title = "Latent GP Realization ($kernel_name)"
    ax.xticks = (1:n_reg_lo + n_reg_hi, regions)
    ax.xticklabelrotation = 45 
    return fig
end

fig2 = plot_gp_aggr(gp_aggr, regions)
save("plots/gp_county_grid.png", fig2)