module Utils 

(export 
dist_euclid, exp_sq_kernel, M_g, M_g, compute_pts_polygons, 
plot_map_with_points, plot_ax_with_points, plot_gp_aggr,
plot_popn_region, compute_pts_in_polys_v3,
vae_encoder, vae_decoder, VAE, loss_function
)
# Imports
using DataFrames: DataFrame
using LinearAlgebra: I
using ArchGDAL: IGeometry, wkbPolygon
import LibGEOS
import GeometryOps
using CairoMakie 
using GeoMakie: GeoAxis, poly!
import ColorSchemes

using Lux
using LuxCore
using ConcreteStructs 
using Random 
#using Zygote 
using MLUtils: rand_like
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

@doc """
Computes if point falls in polygon 
This function also takes into account mutiple polygons corresponding 
to one region. For example data/processed/v3/census/census.shp 
contains multiple polygons for one region (i.e New England containing 5 regions)
But we want all points falling in these 5 regions to account for 1 region (New England)
This is done through looping across all regions and mapping them to 1 region. 
Inputs 
    - coords : lat/lon points as a vector of tuples 
    - df_reg : df_hi or df_lo as a GeoDataFrame 
Outputs 
    - pol_pts : (n_region, n_pts) matrix with a value of 1 if point falls in region else 0
    - pt_which_pol : (n_pts,) vector which contains values 1:n_regions + 1
      (It consists a value of 0 if a point doesnt fall on a region)
"""->
function compute_pts_in_polys_v3(
    coords::Vector{Tuple{Float64,Float64}}, 
    df_reg::DataFrame
)::Tuple{Matrix{Int64}, Vector{Int64}}
    uniq_regs = df_reg.region |> unique
    reg_map = Dict(reg => i for (reg,i) in zip(uniq_regs,range(1,length(uniq_regs))))
    n_pts = length(coords)
    n_pols = length(uniq_regs)
    pol_pts = zeros(Int, (n_pols, n_pts))
    pt_which_pol = zeros(Int, n_pts) # Note a region that isnt included will contain a value 0. 
    for regn in uniq_regs 
        df_flt = filter(:region => x -> x == regn, df_reg)
        geoms = df_flt.geometry # Vector{Geometry}
        for i_pt = 1:n_pts
            pt = coords[i_pt]
            for pol in geoms
                if LibGEOS.contains(pol,pt)
                    i_pol = reg_map[regn] # get numeric region code form mapping
                    pol_pts[i_pol, i_pt] = 1 
                    pt_which_pol[i_pt] = i_pol
                end
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

@doc""" 
Computes the aggraged value mean per region. 
The reason why we do this is when computing just aggregates, the number of
points can scale largely if many points fall on the region. 
When applying logistic at latente prevalence estimation this forms a lot of
extreme values (0,1's) instead of values in between.
Inputs 
    - M : Matrix with binary entries $m_{ij}, $ showing whether point $j$ is in polygon $i$
        - shape : (n_regions, n_grd_pts) ; e.g : (9, 2618)
        - This is the variable pol_pt_lo or pol_pt_hi
    - g : Is a vector of GP draws over the grid
        - shape : (n_grd_pts,) ; e.g (2618,)
        - This is the gp function
    - matmul(M,g) gives a vector sum over each polygon
"""->
function M_g_mean(M,g)
    counts = sum(M; dims = 2) # 9x1 Mat 
    #counts[counts .== 0] .= 1 
    summed = M * g #(9,*)
    mean_vals = summed ./ counts
    return mean_vals
end 

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
Plots a map with Geographic region and points that fall only on a 
specified region. This is used to ensure that lat/lon points have 
indeed been 
Inputs 
    - df_reg : df_census or df_state to plot geometry boundary
    - df_pts : df_county or points dataframe with columns
    poly_hi or poly_lo. These columns can be created using 
    `df_county[!,:poly_lo] = pt_which_pol_lo` which ensures a region 
    index is available every point 
    - reng_idx : Region index (i.e any value between 1:9 for df_census)
Outputs 
    - Makie Figure
"""->
function plot_pts_in_reg(df_reg::DataFrame, df_pts::DataFrame, regn_idx::Int)::Figure
    f = Figure(size = (1200,800));
    us_centroid = GeometryOps.centroid(df_reg.geometry);
    uniq_regs = unique(df_reg.region)
    reg_map = Dict(i => reg for (reg,i) in zip(uniq_regs,range(1,length(uniq_regs))))
    push!(reg_map, 0 => "not inclusive")
    if nrow(df_reg) == 49
        df_flt = filter(:poly_hi => x -> x == regn_idx, df_pts)
        title = "state region : $(reg_map[regn_idx])"
    else
        df_flt = filter(:poly_lo => x -> x == regn_idx, df_pts)
        title = "census region : $(reg_map[regn_idx])"
    end

    ax = GeoAxis(
        f[1,1],
        dest = "+proj=ortho +lon_0=$(us_centroid[1]) +lat_0=$(us_centroid[2])",
        title = title
    );
    poly!(
        ax, 
        df_reg.geometry, 
        color = (:lightgray, 1.0),
        strokecolor = :black, strokewidth = 2
    );
    scatter!(ax, df_flt.centroid_x, df_flt.centroid_y, color = (:blue,1.0), markersize = 7)
    f 
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

# ------------------------------------ VAE ----------------------------------- #
@doc """
Encoder for GP
Inputs
    - rng : Random state 
    - inp_dim : Input Dimension 
    - z_dim : Latent Dimension 
Outputs
    - z : Latent variable z
    - μ : Intermediate computation (Need for KL Divergence)
    - logσ² : Intermediate computation (Needed for KL Divergence)
"""->
function vae_encoder(rng, inp_dim::Int, z_dim::Int)
    return @compact(
        ;
        fc1 = Dense(inp_dim, inp_dim ÷ 2), #(3117,*) -> (1558,*)
        bn1 = BatchNorm(inp_dim ÷ 2),
        fc2 = Dense(inp_dim ÷ 2, inp_dim ÷ 4), #(1558,*) -> (779, *)
        bn2 = BatchNorm(inp_dim ÷ 4),
        fc_mu = Dense(inp_dim ÷ 4, z_dim), #(779, *) -> (389, *)
        fc_logvar = Dense(inp_dim ÷ 4, z_dim)
    ) do x 
    out = elu.(bn1(fc1(x)))
    out = elu.(bn2(fc2(out)))
    μ = fc_mu(out)
    logσ² = fc_logvar(out) # you need to return logσ² as this is whats used in KLDiv
    T = eltype(logσ²)
    σ = exp.(logσ² .* T(0.5))
    ϵ = rand_like(Lux.replicate(rng), σ)
    z = μ .+ σ .* ϵ
    return z, μ, logσ²  
    end
end

@doc """ 
VAE Encoder for AggGP
Inputs 
    - rng : Random state 
    - inp_dim : Input Dimension 
    - h_dim : Hidden Dimension
    - z_dim : Latent Dimension 
Outputs 
    - z : Latent variable z 
    - μ : Intermediate computation (Need for KL Divergence)
    - logσ² : Intermediate computation (Needed for KL Divergence)
"""->
function vae_encoder(rng, inp_dim::Int, h_dim::Int, z_dim::Int)
    return @compact(
        ;
        fc1 = Dense(inp_dim, h_dim), #(58,*) -> (50,*)
        bn1 = BatchNorm(h_dim),
        fc_mu = Dense(h_dim, z_dim), #(50, *) -> (40, *)
        fc_logvar = Dense(h_dim, z_dim) #(50, *) -> (40, *)
    ) do x 
    out = elu.(bn1(fc1(x)))
    μ = fc_mu(out)
    logσ² = fc_logvar(out) # We need this quantity for KLDiv
    T = eltype(logσ²)
    σ = exp.(logσ² .* T(0.5))
    ϵ = rand_like(Lux.replicate(rng), σ)
    z = μ .+ σ .* ϵ 
    return z, μ, logσ²
end
end


@doc """
Decoder for GP
Inputs 
    - z_dim : latent dimension 
    - out_dim : output dimension same as input dimension 
Outputs 
    - gp_recon : Reconstructed GP from Normal(0,1)
"""->
function vae_decoder(z_dim, out_dim)
    return @compact(
        ;
        fc1 = Dense(z_dim, out_dim ÷ 4), #(389, *) -> (779, *)
        bn1 = BatchNorm(out_dim ÷ 4),
        fc2 = Dense(out_dim ÷ 4, out_dim ÷ 2), #(779, *) -> (1558, *)
        bn2 = BatchNorm(out_dim ÷ 2),
        fc3 = Dense(out_dim ÷2, out_dim) #(1558, *) -> (3117, *)
    ) do z 
        out = elu.(bn1(fc1(z))) #(389, *) -> (779, *)
        out = elu.(bn2(fc2(out))) #(779, *) -> (1558, *)
        gp_recon = fc3(out) #(1558, *) -> (3117, *)
        @return gp_recon
end
end

@doc"""
VAE Decoder for AggGP
Inputs 
    - z_dim : latent dimension from which gp is reconstructed 
    - h_dim : hidden dimension 
    - out_dim : Output dimension which is the same as encoders input dimension
Outputs 
    - gp_recon : Reconstructed GP 
"""->
function vae_decoder(z_dim, h_dim, out_dim)
    return @compact(
        ;
        fc1 = Dense(z_dim, h_dim),
        bn1 = BatchNorm(h_dim),
        fc2 = Dense(h_dim, out_dim)
    ) do z 
    out = elu.(bn1(fc1(z)))
    gp_recon = fc2(out)
    return gp_recon
end 
end

@concrete struct VAE <: AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder <: AbstractLuxLayer 
    decoder <: AbstractLuxLayer  
end

@doc """
VAE model comprising of encoder and decoder (For GP)
Inputs 
    - inp_dim : input dimension 
    - z_dim : latent dimension 
Output 
    - VAE : Variational AutoEncoder struct 
"""-> 
function VAE(rng, inp_dim, z_dim)
    encoder = vae_encoder(rng, inp_dim, z_dim)
    decoder = vae_decoder(z_dim, inp_dim) # second argument is out_dim = inp_dim
    return VAE(encoder, decoder)
end

@doc """
VAE model comprising of the encoder and decoder (For Agg GP)
Inputs 
    - inp_dim : input dimension 
    - z_dim : latent dimension 
Output 
    - VAE : Variational Autoencoder struct 
"""->
function VAE(rng, inp_dim, h_dim, z_dim)
    encoder = vae_encoder(rng, inp_dim, h_dim, z_dim)
    decoder = vae_decoder(z_dim, h_dim, inp_dim) #inp_dim same as out_dim
    return VAE(encoder, decoder)
end

# forward function 
@doc """
Forward method 
Inputs 
    - x : GP 
    - ps : VAE params 
    - st : VAE states
"""->
function (vae::VAE)(x, ps,st)
    (z,μ,logσ²), st_enc = vae.encoder(x, ps.encoder, st.encoder)
    gp_recon, st_dec = vae.decoder(z, ps.decoder, st.decoder)
    return (gp_recon, μ, logσ²), (;encoder = st_enc, decoder = st_dec)
end

@doc """
Reconstruction Loss & KL Divergence Loss 
Inputs 
    - model : VAE 
    - ps : VAE params (both encoder and decoder)
    - st : VAE states (both encoder and decoder)
    - X : Gaussian Process (3117, *)
"""
function loss_function(model, ps, st, X)
    (X_recon, μ, logσ²), st = model(X, ps, st) #(3117,*),(389,*),(389,*)
    recon_loss = MSELoss(agg = sum)(X_recon, X) #(,) <- float  
    kldiv_loss = -sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²)) / 2 #(,) <- float
    loss = recon_loss + kldiv_loss 
    return loss, st, (;X_recon, μ, logσ², recon_loss, kldiv_loss)
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




