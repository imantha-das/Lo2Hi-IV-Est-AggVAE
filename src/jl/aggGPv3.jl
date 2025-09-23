begin
    import Pkg
    Pkg.activate("jl_env/lo2hi")
    import CSV
    using DataFrames 
    import GeoDataFrames 
    import LibGEOS
    include("utils.jl")
    (using .Utils: 
    compute_pts_polygons, M_g,
    plot_map_with_points,plot_ax_with_points, plot_gp_aggr,
    plot_popn_region, plot_gps
    )
    using AbstractGPs: GP, SEKernel, GibbsKernel
    using StaticArrays: SVector
    using CairoMakie
    using GeoMakie:GeoAxis, poly!
    import GeometryOps
    using Turing
    using Turing:logistic
    set_theme!(theme_light())
end


#if abspath(PROGRAM_FILE) == @__FILE__ 
# ------------------------------------ GiS ----------------------------------- #
# Load county : Grid Points will be at this level
county_fold = "data/processed/gis/county_grid_pop_v2"
county_files = filter(x -> endswith(x, ".shp"), readdir(county_fold))
county_path = joinpath(county_fold, first(county_files))
df_county = GeoDataFrames.read(county_path)

# Load Census : Low Resolution data
census_fold = "data/processed/gis/low/us_census_divisions"
census_files = filter(x -> endswith(x,".shp"), readdir(census_fold)) 
census_path = joinpath(census_fold, first(census_files))
df_census = GeoDataFrames.read(census_path)
rename!(df_census, :area => :region)
select!(df_census, :geometry, :region)

# Load States : High resolution data 
state_fold = "data/processed/gis/high/us_state_divisions"
state_files = filter(x -> endswith(x, ".shp"), readdir(state_fold))
state_path = joinpath(state_fold, state_files[1])
df_state = GeoDataFrames.read(state_path)
rename!(df_state, :area => :region)
select!(df_state, :geometry, :region)

# -------------------------------- Population -------------------------------- #
df_census_pop = CSV.read("data/processed/gis/low/census_popn.csv", DataFrame)
df_state_pop = CSV.read("data/processed/gis/high/state_popn.csv", DataFrame)
rename!(df_census_pop, :division => :region)
rename!(df_state_pop, :state => :region)

# --------------------------------- Influenza -------------------------------- #
infz_data_root = "data/processed/fluview/nrevss_phl_20152024"
df_infz_census_1524 = CSV.read(joinpath(infz_data_root, "infz_census_20152024.csv"), DataFrame)
df_infz_state_1524 = CSV.read(joinpath(infz_data_root, "infz_state_20152024.csv"), DataFrame)
year = 2018 #todo pass as an argparse argument 
df_infz_census = filter(:year => ==(year), df_infz_census_1524)
df_infz_state = filter(:year => ==(year), df_infz_state_1524)
select!(df_infz_census, :region, :year, :test_cases, :tested_pos)
select!(df_infz_state, :region, :year, :test_cases, :tested_pos)

# ------------------------ Join Influzenz and Gis Data ----------------------- #
df_lo = innerjoin(df_census, df_infz_census, df_census_pop, on = :region)
df_hi = innerjoin(df_state, df_infz_state, df_state_pop, on = :region)

fig1 = plot_popn_region(
    df_lo.geometry, df_hi.geometry, 
    df_lo.tot_popn, df_hi.tot_popn, 
    df_lo.region, df_hi.region,
    year
)

# ---------------------------- Polygons and Points --------------------------- #
coords = zip(df_county.centroid_x, df_county.centroid_y) |> collect
county_polys = df_county.geometry
census_polys = df_lo.geometry
state_polys = df_hi.geometry
pol_pts_lo, pt_which_pol_lo = compute_pts_polygons(coords, census_polys)
pol_pts_hi, pt_which_pol_hi = compute_pts_polygons(coords, state_polys)

pts = hcat(df_county.centroid_x, df_county.centroid_y) 

fig2 = plot_map_with_points(
    census_polys, 
    pts, 
    pt_which_pol_lo, 
    "GP evaluating points at census level",
    col_by_reg = false
)

fig3 = Figure(size = (1200,800), title = "Grid Points at County Level");

us_centroid = GeometryOps.centroid(census_polys)
ax_lo = GeoAxis(
    fig3[1,1],
    dest = "+proj=ortho +lon_0=$(us_centroid[1]) +lat_0=$(us_centroid[2])",
    title = "GP evaluation points for Census regions"
)
plot_ax_with_points(ax_lo,census_polys,pts,pt_which_pol_lo,:dodgerblue2)

us_centroid = GeometryOps.centroid(state_polys)
ax_hi = GeoAxis(
    fig3[1,2],
    dest = "+proj=ortho +lon_0=$(us_centroid[1]) +lat_0=$(us_centroid[2])",
    title = "GP evaluation points for State regions"
)
plot_ax_with_points(ax_hi,state_polys, pts, pt_which_pol_hi, :darkolivegreen)
fig3
save("plots/county_grids.png",fig3)

# ----------------------------- Population Matrix ---------------------------- #
x_pop = df_county[!, :pop2021]
x_pop_mat = x_pop ./ sum(x_pop)
polys = df_county.geometry

f = Figure(size = (1200,800))
us_centroid = GeometryOps.centroid(polys)
ax = GeoAxis(
    f[1,1],
    dest = "+proj=ortho +lon_0=$(us_centroid[1]) +lat_0=$(us_centroid[2])",
    title = "County level population"
)
poly!(
    ax, 
    polys, 
    color = (:lightgray, 1.0)
)
CairoMakie.Colorbar(
    f[1,2], 
    limits = extrema(log.(x_pop_mat .+ 1e-6)),
    label = "log.(pₓ/Σpₓ)",
    colormap = :viridis
)
sizes = 4 .+ 30 .* (x_pop_mat ./ maximum(x_pop_mat))
scatter!(
    ax, pts[:,1], pts[:,2], 
    markersize = sizes, 
    color = log.(x_pop_mat .+ 1e-6), 
    colormap = :viridis, 
    colorrange = extrema(log.(x_pop_mat .+ 1e-6))
)
f

# ---------------------------- Gaussian Processes ---------------------------- #

x = copy(pts) #(3120,2)
k = SEKernel()
f = GP(k)
x_svec = [SVector{2,}(x[i,:]) for i in 1:size(x,1)]
f_latent = f(x_svec, 1e-4)
f_reals = rand(f_latent, 100)

fig4 = plot_gps(f_reals)
save("plots/gp_at_grid.png", fig4)

lo_regions = df_lo.region |> collect
hi_regions = df_hi.region |> collect
regions = vcat(lo_regions, hi_regions)

gp_aggr_lo = M_g(pol_pts_lo, rand(f_latent, 100)) #(n_reg_lo,n_samps)
gp_aggr_hi = M_g(pol_pts_hi, rand(f_latent, 100)) #(n_reg_hi,n_samps)
gp_aggr = vcat(gp_aggr_lo,gp_aggr_hi) #(n_reg_lo+hi, n_samps)

n_samples = 100
μp = rand(Normal(0,1), n_samples)
p_logits = reshape(μp, (1,100)) .+ gp_aggr 
p = logistic.(p_logits)
fig5= plot_gp_aggr(p, regions, title = "Aggregated Gps")

p_logits_lo = reshape(μp, (1,n_samples)) .+ gp_aggr_lo 
p_logits_hi = reshape(μp, (1,n_samples)) .+ gp_aggr_hi
p_lo = logistic.(p_logits_lo) 
p_hi = logistic.(p_logits_hi)
fig_51 = plot_gp_aggr(vcat(p_lo,p_hi), regions, title = "Aggreated Gps, Normalised seperately")
fig_51
#save("plots/gp_county_grid.png", fig5)

# ----------------------------- Agg Prev Process ----------------------------- #
n_samples = 100
μp = rand(Normal(0,1), n_samples)
p_logits =  reshape(μp, (1,100)) .+ f_reals #(3107,n_samples)
p = logistic.(p_logits)
p_lo = M_g(pol_pts_lo, p) #(9,n_samples)
p_hi = M_g(pol_pts_hi, p) #(49,n_samples)
fig6 = plot_gp_aggr(vcat(p_lo,p_hi),regions, title = "Prevelance Prior Predictive Dist : σ(μ_prev + gp) * M_region_agg")
plot_gp_aggr(logistic.(vcat(p_lo,p_hi)), regions, title = "Prevelance Prior logistic")
save("plots/prev_prior_pred_dist_popweighted.png",fig7)
p_lo[:,1] |> sum

# --------------------------- Population weighting --------------------------- #
n_samples = 100
μp = rand(Normal(0,1), n_samples)
p_logits =  reshape(μp, (1,100)) .+ f_reals #(3107,n_samples)
p = logistic.(p_logits)
p_weighted = p .* x_pop_mat
p_lo = M_g(pol_pts_lo, p_weighted) #(9,n_samples)
p_hi = M_g(pol_pts_hi, p_weighted) #(49,n_samples)
fig7 = plot_gp_aggr(vcat(p_lo,p_hi),regions, title = "Prevelance Prior Predictive Dist Weighted : σ(μ_prev + gp) * M_region_agg")
plot_gp_aggr(logistic.(vcat(p_lo,p_hi)), regions, title = "Prevelance Prior logistic")

# --------------------------------- Infz Data -------------------------------- #

n_tested_lo = Vector{Int64}(df_lo[!,:test_cases])
n_tested_hi = Vector{Int64}(df_hi[!,:test_cases])
n_positive_lo = Vector{Int64}(df_lo[!,:tested_pos])
n_positive_hi = Vector{Int64}(df_hi[!, :tested_pos])

fig4 = Figure()
ax4 = Axis(fig4[1,1])
barplot!(ax4, vcat(n_tested_lo, n_tested_hi), label = "num_tested")
barplot!(ax4, vcat(n_positive_lo,n_positive_hi), label = "num_tested_positive")
axislegend(ax4,position = :rt)
ax4.xticks = (1:length(regions), regions)
ax4.xticklabelrotation = 45
fig4

# ------------------------------ Binomial Models ----------------------------- #

@model function prev_gp_aggr_bin(
    x::Matrix{Float64},
    x_pop::Vector{Float64},
    M_lo::Matrix{Int64},
    M_hi::Matrix{Int64},
    n_tested_lo::Vector{Int64},
    n_tested_hi::Vector{Int64},
    n_positive_lo::Vector{Int64};
    jitter::Float64 = 1e-4
)
    # Compute GP realization
    x_svec = [SVector{2,}(x[i,:]) for i in 1:size(x,1)] #3107 - Vector{Svector{2,}}
    k = SEKernel()
    f = GP(k)
    f_latent ~ f(x_svec, jitter) #(n_pts,)
    # Prevalence model 
    μp ~ Normal(0,1) # mean prevalence 
    logits = @. μp + f_latent # (n_pts,) linear predictor for prevalence 
    p = logistic.(logits)
    # Weighted prevalence 
    pw = p .* x_pop
    # Aggregate based on points falling on regions 
    #* We are aggragting prevalence instead of gp
    p_aggr_lo = M_g(M_lo, pw)
    p_aggr_hi = M_g(M_hi, pw)
    p_aggr = vcat(p_aggr_lo, p_aggr_hi) 

    n_tested = vcat(n_tested_lo, n_tested_hi) #(58,)
    n_regions_lo = size(n_tested_lo,1) # 9
    n_regions_hi = size(n_tested_hi,1) # 49
    n_regions = n_regions_lo + n_regions_hi
    n_positive = vcat(n_positive_lo, fill(missing, n_regions_hi)) #(58,)
    for i = 1:n_regions
        n_positive[i] ~ Binomial(n_tested[i], p_aggr[i]) 
    end
    return (
        fx = f(x_svec, jitter),
        f_latent = f_latent,
        p_aggr = p_aggr,
        n_postive = n_positive,
    )
end

@model function prev_gp_aggr_betabin(
    x::Matrix{Float64},
    x_pop::Vector{Float64},
    M_lo::Matrix{Int64},
    M_hi::Matrix{Int64},
    n_tested_lo::Vector{Int64},
    n_tested_hi::Vector{Int64},
    n_positive_lo::Vector{Int64};
    jitter::Float64 = 1e-4
)
    # GP realization
    x_svec = [SVector{2,}(x[i,:]) for i in 1:size(x,1)]
    k = SEKernel()
    f = GP(k)
    f_latent ~ f(x_svec, jitter) # (n_pts,)

    # Prevalence model
    μp ~ Normal(0, 1)             # mean prevalence
    logits = @. μp + f_latent
    p = logistic.(logits)         # (n_pts,)

    # Overdispersion for Beta-Binomial
    ϕ ~ Exponential(1.0)          # phi > 0

    # Weighted prevalence
    pw = p .* x_pop

    # Aggregate based on points falling on regions
    p_aggr_lo = M_g(M_lo, pw)
    p_aggr_hi = M_g(M_hi, pw)
    p_aggr = vcat(p_aggr_lo, p_aggr_hi)

    n_tested = vcat(n_tested_lo, n_tested_hi)
    n_regions_lo = size(n_tested_lo, 1)
    n_regions_hi = size(n_tested_hi, 1)
    n_regions = n_regions_lo + n_regions_hi
    n_positive = vcat(n_positive_lo, fill(missing, n_regions_hi))

    for i = 1:n_regions
        α = p_aggr[i] * ϕ + 1e-5  # small jitter to avoid zero
        β = (1 - p_aggr[i]) * ϕ + 1e-5
        n_positive[i] ~ BetaBinomial(n_tested[i], α, β)
    end

    return (
        fx = f(x_svec, jitter),
        f_latent = f_latent,
        p_aggr = p_aggr,
        n_positive = n_positive,
        phi = ϕ
    )
end

@model function prev_gp_aggr_betabin_v2(
    x::Matrix{Float64},
    x_pop::Vector{Float64},
    M_lo::Matrix{Int64},
    M_hi::Matrix{Int64},
    n_tested_lo::Vector{Int64},
    n_tested_hi::Vector{Int64},
    n_positive_lo::Vector{Int64};
    jitter::Float64 = 1e-4
)
    # GP realization
    x_svec = [SVector{2,}(x[i,:]) for i in 1:size(x,1)]
    k = SEKernel()
    f = GP(k)
    f_latent ~ f(x_svec, jitter) # (n_pts,)

    # Prevalence model
    μp ~ Normal(0, 1)             # mean prevalence
    logits = @. μp + f_latent
    p = logistic.(logits)         # (n_pts,)

    # Overdispersion for Beta-Binomial
    ϕ ~ Exponential(1.0)          # phi > 0

    # Weighted prevalence
    pw = p .* x_pop

    # Aggregate based on points falling on regions
    p_aggr_lo = M_g(M_lo, pw)
    p_aggr_hi = M_g(M_hi, pw)

    n_regions_lo = size(n_tested_lo, 1)
    n_regions_hi = size(n_tested_hi, 1)

    for i = 1:n_regions_lo
        α = p_aggr_lo[i] * ϕ + 1e-5  # small jitter to avoid zero
        β = (1 - p_aggr_lo[i]) * ϕ + 1e-5
        n_positive_lo[i] ~ BetaBinomial(n_tested_lo[i], α, β)
    end

    # SInce we are creating this vector inside Turing will not condition on them 
    n_positive_hi = Vector{Int}(undef, length(n_tested_hi))
    for i = 1:n_regions_hi 
        α = p_aggr_hi[i] * ϕ + 1e-5 
        β = (1 - p_aggr_hi[i]) * ϕ + 1e-5
        n_positive_hi[i] ~ BetaBinomial(n_tested_hi[i], α, β)
    end

    return (
        fx = f(x_svec, jitter),
        f_latent = f_latent,
        p_aggr_lo = p_aggr_lo,
        p_aggr_hi = p_aggr_hi,
        n_positive_lo = n_positive_lo,
        n_positive_hi = n_positive_hi,
        phi = ϕ
    )
end


model = prev_gp_aggr_betabin_v2(
    x,
    x_pop_mat,
    pol_pts_lo,
    pol_pts_hi,
    n_tested_lo,
    n_tested_hi,
    n_positive_lo
)
posterior_samples = sample(model, NUTS(5,0.65), 100)
pos_df = posterior_samples |> DataFrame
pos_df."f_latent[59]"
pos_df."n_positive[3]"

# -------------------------------- Predictions ------------------------------- #
