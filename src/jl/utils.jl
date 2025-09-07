module Utils 

(export 
dist_euclid, exp_sq_kernel, M_g, compute_pts_polygons, 
plot_map_with_points, plot_ax_with_points, plot_gp_aggr,
plot_popn_region
)
# Imports
using LinearAlgebra: I
using ArchGDAL: IGeometry, wkbPolygon
import LibGEOS
import GeometryOps
using CairoMakie 
using GeoMakie: GeoAxis, poly!
import ColorSchemes
set_theme!(theme_light())

# ----------------------- Computational Grid Functions ----------------------- #
# Eucledian Distance
@doc """
Computes the eucledian distance between regions, this funvtion
is used to find the distance between points.
Inputs
    - x : lat/lon values
    - z : lat/lon values
Note for this problem x and z are the same as we are trying to find
the distance between each point and all other points.
"""->
function dist_euclid(x::Array{Float64},z::Array{Float64})::Array{Float64}
    # If x or z is in the shape (n_grid_pts,) -> (n_grid_pts, 1)
    if ndims(x) == 1
        x = reshape(x, (length(x), 1)) # if (n_grd_pts,) -> (n_grd_pts, 1)
    end
    if ndims(z) == 1
        z = reshape(z, (length(z), 1))
    end
    # number of grid pints and lat/lon dims
    n_x, m = size(x) # n_grd_pts , 2
    n_z, m_z = size(z)
    @assert m == m_z
    # contruct a matrix wuth shape (n_grd_pts, n_grd_pts)
    delta = zeros((n_x,n_z)) #(n_grd_pts, n_grd_pts)
    for d = 1:m
        # in the first iter they consider lat in 2nd ite they consider lon
        x_d = x[:,d] #(2618,)
        z_d = z[:,d] #(2618,)
        # compute distance between each point and all other points
        delta .+= (x_d .- reshape(z_d, (1, length(z_d)))).^2
    end
    return sqrt.(delta) #(2618,2618)
end

# Exponential Square Kernel
@doc """
Exponential Square Kernel
Inputs
    - x : Lat/Lon for Grid
    - z : Lat/Lon for Grid
    For this problem x and z are the same 
"""->
function exp_sq_kernel(x::Matrix{Float64},z::Matrix{Float64},k_var::Float64,k_length::Float64,noise::Float64,jitter::Float64=1.0e-4)::Matrix{Float64}
    dist = dist_euclid(x,z) #(n_grd_pts,n_grid_pts)
    deltaXsq = (dist ./ k_length).^2 #(n_grd_pts, n_grd_pts)
    k = k_var .* exp.(-0.5 .* deltaXsq) #(n_grd_pts, n_grd_pts)
    k .+= (noise + jitter) .* Matrix(I, (size(x)[1],size(x)[1])) #(n_grd_pts, n_grd_pts)
    return k #(n_grd_pts, n_grd_pts)
end

@doc """
Computes which points fall into which polygons 
Inputs 
    - coords : Lat/Lon value of points 
    - poly_regions : Regions expressed as geometry object
Outputs
    - pol_pts : A 
"""->
function compute_pts_polygons(coords, poly_regions) 
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

# Aggregate points in region func
@doc"""
Used to aggregated values per region 
Inputs 
    - M : Matrix with binary entries $m_{ij}, $ showing whether point $j$ is in polygon $i$
        - shape : (n_regions, n_grd_pts) ; e.g : (9, 2618)
        - This is the variable pol_pt_lo or pol_pt_hi
    - g : Is a vector of GP draws over the grid
        - shape : (n_grd_pts,) ; e.g (2618,)
        - This is the gp function
    - matmul(M,g) gives a vector sum over each polygon
"""->
M_g(M,g) = M * g #(n_regions,) e.g (9,)

# -------------------------- Visialization Functions ------------------------- #
@doc """
Plots a Map with the computational grid 
Inputs 
    - polys : polygons of regions, comes as geometry from geospatial packages such as GeoDataFrames
    - pts : Lat/Lon values of the computational grid points 
    - pt_which_pol : Int vector conaining a value for the regions each is in point (1..n_regions)
    - title : title for the figure 
    - col_by_reg : boolean to color by region 
Outputs 
    - Makie Figure 
"""->
function plot_map_with_points(polys, pts::Matrix, pt_which_pol::Vector{Int64}, title::String; col_by_reg::Bool = false)
    f = Figure(size = (1200,800))
    us_centroid = GeometryOps.centroid(polys)
    ax = GeoAxis(
        f[1,1],
        dest = "+proj=ortho +lon_0=$(us_centroid[1]) +lat_0=$(us_centroid[2])",
        title = title
    )
    poly!(
        ax, 
        polys, 
        color = (:lightblue, 0.5),strokecolor = :black, strokewidth = 2
    )
    if col_by_reg
        cmap = ColorSchemes.tab20.colors  # categorical colormap
        # pt_which_pol is vector numerical integer for every point
        # where these numerical integers will be of 1:n_regions
        colors = [cmap[(r % length(cmap)) + 1] for r in pt_which_pol]
        # Plot points with region-based colors
        scatter!(ax, pts[:,1], pts[:,2], markersize = 4, color = colors)
    else
        scatter!(ax, pts[:,1], pts[:,2], markersize = 4, color = (:red))
    end
    return f
end

@doc """
Same as `plot_map_with_points` but inputs an ax, useful for creating
subplots 
"""->
function plot_ax_with_points(ax, polys, pts::Matrix, pt_which_pol::Vector{Int64}, color::Symbol; col_by_reg::Bool = false)
    poly!(
        ax, 
        polys, 
        color = (:lightblue, 0.5),strokecolor = :black, strokewidth = 2
    )
    if col_by_reg
        cmap = ColorSchemes.tab20.colors  # categorical colormap
        # pt_which_pol is vector numerical integer for every point
        # where these numerical integers will be of 1:n_regions
        colors = [cmap[(r % length(cmap)) + 1] for r in pt_which_pol]
        # Plot points with region-based colors
        scatter!(ax, pts[:,1], pts[:,2], markersize = 4, color = colors)
    else
        scatter!(ax, pts[:,1], pts[:,2], markersize = 4, color = color)
    end
    return ax
end

@doc """
Plots an aggregated GP 
Inputs 
- gp_aggr : aggregated gp with shape (n_lo+n_hi,)
- regions : String values of regions to be used as xticks 
- n_reg_lo : Number of regions in low resolution administrative boundaries (i.e 9 census regions)
- n_reg_hi : Number of regions in high resolution administrative boundaries (i.e 49 state regions)
- kernel_name : Name of kernel (i.e RBF) to be used in the title
"""->
function plot_gp_aggr(gp_aggr::Matrix{Float64}, regions::Vector{String};n_reg_lo::Int = 9, n_reg_hi::Int = 49, title::String = "Aggregated Latent GP Realizations (RBF)")
    fig = Figure(size = (1200,400), )
    ax = Axis(fig[1,1])
    for i = 1:size(gp_aggr,2)
        lines!(ax,1:size(gp_aggr,1), gp_aggr[:,i], color = (:black,0.3))
    end
    vspan!(ax, [0],[n_reg_lo], color =(:dodgerblue2, 0.3), label = "low resolution")
    vspan!(ax, [n_reg_lo], [n_reg_lo + n_reg_hi], color = (:darkolivegreen,0.2),label = "high resolution")
    axislegend(positon = :rt)
    ax.title = title
    ax.xticks = (1:n_reg_lo + n_reg_hi, regions)
    ax.xticklabelrotation = 45 
    return fig
end

@doc """
Plot population by region 
Inputs
    - census/state_polys : census/state level geometries
    - census/state_pop : census/state population
    - region_lo or region_hi : region names 
    - year : year the population was taken
"""->
function plot_popn_region(census_polys, state_polys, census_pop, state_pop, regions_lo, regions_hi, year)
    fig = Figure(size = (1200,800))
    census_centroid = GeometryOps.centroid(census_polys)
    ax1 = GeoAxis(
        fig[1,1],
        dest ="+proj=ortho +lon_0=$(census_centroid[1]) +lat_0=$(census_centroid[2])",
        title = "Census population"
    )
    po1 = poly!(
        ax1,
        census_polys,
        color = census_pop,
        strokecolor = :black, strokewidth = 2
    )
    Colorbar(fig[1,2], po1)

    ax2 = GeoAxis(
        fig[1,3],
        dest ="+proj=ortho +lon_0=$(census_centroid[1]) +lat_0=$(census_centroid[2])",
        title = "State population"
    )
    po2 = poly!(
        ax2,
        state_polys,
        color = state_pop,
        strokecolor = :black, strokewidth = 2
    )
    Colorbar(fig[1,4], po2)

    ax3 = Axis(
        fig[2,1:4], title = "Combined Census and State Population for $year "
    )
    regions = vcat(regions_lo, regions_hi)
    ax3.xticks = (1:length(regions), regions)
    ax3.xticklabelrotation = 45 
    barplot!(ax3,vcat(census_pop,state_pop))
    return fig
end

@doc """
Plot raw gps, not the one aggreagted
"""->
function plot_gps(gps)
    n_samples = size(gps,2)
    fig = Figure()
    ax = Axis(fig[1,1], title = "Gp at each grid point")
    for i in 1:n_samples
        lines!(ax,1:size(gps,1),gps[:,i], color = (:black, 0.2))
    end 
    return fig 
end

# ----------------------------------- Main ----------------------------------- #
# Main - Just to test the Utils module
if abspath(PROGRAM_FILE) == @__FILE__
    using .Utils
    x = rand(2618, 2) # random lat/lon values
    @show size(x)

    dist = Utils.dist_euclid(x,x)
    @show size(dist)

    k = Utils.exp_sq_kernel(x, x, 1.0, 1.0, 1e-4)
    @show size(k)

    pol_pt_lo = Int.(rand(Bool, 9, 2618)) # random binary matrix for low polygon points
    f = rand(2618) # random GP function values
    aggregated_lo = Utils.M_g(pol_pt_lo, f)
    @show size(aggregated_lo)
    
end
end




