import Pkg 
Pkg.activate("lo2hi")
include("Utils.jl")
using .Utils: dist_euclid, exp_sq_kernel, M_g
using NPZ: npzread
import GeoIO 
using CairoMakie 
using AbstractGPs: GP, SEKernel, with_lengthscale, Kernel, SqExponentialKernel
#using KernelFunctions: SquareExponentialKernel
using Turing
using DataFrames: DataFrame, names, completecases
using StaticArrays

begin 
    data_root = "data/processed"
    # lat/lon values of artficial grid
    x = npzread(joinpath(data_root, "lat_lon_x.npy")) #(2618,2)
    # Coarse/Low resolution administrative region
    # Points that fall on the region takes a value 1 else 0
    pol_pt_lo = npzread(joinpath(data_root, "low", "pol_pt_lo.npy")) #(9,2618)
    # Points that fall on the region takes values 0::num_regions
    pt_which_pol_lo = npzread(joinpath(data_root, "low", "pt_which_pol_lo.npy")) #(2618,)
    # High resolution administrative num_regions
    pol_pt_hi = npzread(joinpath(data_root, "high", "pol_pt_hi.npy"))
    pt_which_pol_hi = npzread(joinpath(data_root, "high", "pt_which_pol_hi.npy"))
    # Geopandas dataframes
    df_lo = GeoIO.load(joinpath(data_root, "low", "us_census_divisions", "us_census_divisions.shp"))
    df_hi = GeoIO.load(joinpath(data_root, "high", "us_state_divisions", "us_state_divisions.shp"))
    # Number of tested cases - This is "n" in Binomial(n,p)
    # You need to add Vector{Int64} or use skipmissing to get rid of type Vector{Union{Missing, Int64}}
    n_tested_lo = Vector{Int64}(df_lo[:, "tot_specs"])
    n_tested_hi = Vector{Int64}(df_hi[:, "tot_specs"])
    # Get number tested positive - This is "y" ~ Binomial(n,p)
    n_positive_lo =Vector{Int64}(df_lo[:, "tot_cases"])
    # Note "p" is prevelence which we model as linear function with Random effects GP
    n_positive_hi = Vector{Int64}(df_hi[:, "tot_cases"])
end

@model function gp_aggr(
    x::Matrix{Float64},
    M_lo::Matrix{Int64},
    M_hi::Matrix{Int64},
    jitter::Float64 = 1e-4,
    kernel::Kernel = SEKernel()
)   
    # x is a matrix of shape (2618,2) with lat/lon values
    # but what we want is Vector{Vector{(1,2)} x 2618}
    #x_svec =  [x[i,:] for i in 1:size(x,1)]
    grd_pts, dims = size(x)
    x_svec = [SVector{dims,}(x[i,:]) for i in 1:size(x,1)]
    ℓ ~ InverseGamma(3,3) # greek letter "ell", kernel length
    σ ~ truncated(Normal(0,0.5)) # kernel variance
    k = σ^2 * with_lengthscale(kernel, ℓ) # Exponential Square Kernel
    f = GP(k)
    f_latent ~ f(x_svec, jitter) # GP latent function values

    # Aggregated results per region. Matrix multiplication of GP 
    # and Grid points that is contained in the region (depicted by value 1)
    # f_latent returns a vector{Real} so convert it to Float64
    gp_aggr_lo = M_g(M_lo, Float64.(f_latent)) # (n_regions_lo,) 
    gp_aggr_hi = M_g(M_hi, Float64.(f_latent)) # (n_regions_hi,)
    return (
        fx = f(x_svec, jitter),
        f_latent = f_latent,
        gp_aggr_lo = gp_aggr_lo,
        gp_aggr_hi = gp_aggr_hi
    )
end

@model function gp_aggrv2(
    x,
    M_lo,
    M_hi,
    jitter = 1e-4,
    kernel= SEKernel()
)   
    # x is a matrix of shape (2618,2) with lat/lon values
    # but what we want is Vector{Vector{(1,2)} x 2618}
    #x_svec =  [x[i,:] for i in 1:size(x,1)]
    grd_pts, dims = size(x)
    x_svec = [SVector{dims,}(x[i,:]) for i in 1:size(x,1)]
    #ℓ ~ InverseGamma(3,3) # greek letter "ell", kernel length
    #σ ~ truncated(Normal(0,0.5)) # kernel variance
    #k = σ^2 * with_lengthscale(kernel, ℓ) # Exponential Square Kernel
    k = kernel
    f = GP(k)
    f_latent ~ f(x_svec, jitter) # GP latent function values

    # Aggregated results per region. Matrix multiplication of GP 
    # and Grid points that is contained in the region (depicted by value 1)
    # f_latent returns a vector{Real} so convert it to Float64
    gp_aggr_lo = M_g(M_lo, Float64.(f_latent)) # (n_regions_lo,) 
    gp_aggr_hi = M_g(M_hi, Float64.(f_latent)) # (n_regions_hi,)
    return (
        fx = f(x_svec, jitter),
        f_latent = f_latent,
        gp_aggr_lo = gp_aggr_lo,
        gp_aggr_hi = gp_aggr_hi
    )
end


@model function prev_gp_aggr(
    x::Matrix{Float64},
    M_lo::Matrix{Int64},
    M_hi::Matrix{Int64},
    n_tested_lo::Vector{Int64},
    n_tested_hi::Vector{Int64},
    n_positive_lo::Vector{Int64}, # This is y 
    jitter::Float64 = 1e-4
)   
    # x is a matrix of shape (2618,2) with lat/lon values
    # but what we want is Vector{Vector{(1,2)} x 2618}
    _,dims = size(x)
    x_svec = [SVector{dims}(x[i,:]) for i in 1:size(x,1)]
    ℓ ~ InverseGamma(3,3) # greek letter "ell", kernel length
    σ ~ truncated(Normal(0,0.5)) # kernel variance
    k = σ^2 * with_lengthscale(SEKernel(), ℓ) # Exponential Square Kernel
    f = GP(k)
    f_latent ~ f(x_svec, jitter) # GP latent function values

    # Aggregated results per region. Matrix multiplication of GP 
    # and Grid points that is contained in the region (depicted by value 1)
    # f_latent returns a vector{Real} so convert it to Float64
    gp_aggr_lo = M_g(M_lo, f_latent) # (n_regions_lo,) 
    gp_aggr_hi = M_g(M_hi, f_latent) # (n_regions_hi,)
    gp_aggr = vcat(gp_aggr_lo, gp_aggr_hi) # (n_regions_lo + n_regions_hi,)

    # Fixed effects : Represents the mean prevalence probibility
    b0 ~ Normal(0,1) # Intercept
    # We will model prevelence as a linear function of Fixed and Random effects 
    lp = @. b0 + gp_aggr # Linear predictor (n_regions_lo + n_regions_hi,)
    # Tested cases are the "n" in Binomial(n,p) 
    n_tested = vcat(n_tested_lo, n_tested_hi) # (n_regions_lo + n_regions_hi,)
    
    n_regions_hi = size(M_hi)[1] # Number of high resolution regions
    n_regions_lo = size(M_lo)[1] # Number of low resolution regions
    n_positive = vcat(n_positive_lo, fill(missing, n_regions_hi)) # (n_regions_lo + n_regions_hi,)
    
    for i in 1:(n_regions_lo + n_regions_hi)
        if !ismissing(n_positive[i])
            # We model the prevelence as Binomial(n,p) where p is the prevelence
            n_positive[i] ~ BinomialLogit(n_tested[i], lp[i])
        end
    end
    return (
        fx = f(x_svec, jitter),
        f_latent = f_latent,
        gp_aggr_lo = gp_aggr_lo,
        gp_aggr_hi = gp_aggr_hi,
        lp = lp,
        n_positive = n_positive
    )
end

function plot_gp_aggr(chn::Chains,M_lo::Matrix{Int64}, M_hi::Matrix{Int64}, n_samples::Int64)
    set_theme!(theme_light())
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], title = "GP Aggregation")
    for s = 1:n_samples
        gp_per_sample = [chn["f_latent[$i]"][s] for i in 1:2618]
        fl_lo = M_g(M_lo, gp_per_sample)
        fl_hi= M_g(M_hi, gp_per_sample)
        #@show size(gp_aggr_sample)
        lines!(ax, 1:length(fl_lo), fl_lo, color = (:blue, 0.3)) 
        lines!(ax, (length(fl_lo) + 1):(length(fl_lo) + length(fl_hi)), fl_hi, color = (:red, 0.3))
    end
    return fig
end


#if abspath(PROGRAM_FILE) == @__FILE__
# Test if exported dunction work as expected 
# k = exp_sq_kernel(x,x, 1.0, 1.0, 1e-4)
# @show size(k)
#m_gp_aggr = gp_aggr(x, pol_pt_lo, pol_pt_hi, 1e-4)
# accessing latent variable 
#@show size(m_gp_aggr().f_latent) 
# Prior Predicitve simulation for gp_aggr func
#m_aggr_chn = sample(m_gp_aggr, Prior(), 100) 
#m_aggr_df = DataFrame(m_aggr)
#@show m_aggr["gp_aggr_lo"]

#fig = plot_gp_aggr(m_aggr_chn, pol_pt_lo, pol_pt_hi, 100)

# We will be testing the prevelence model 
# m_aggr_prev = prev_gp_aggr(
#     x, 
#     pol_pt_lo, pol_pt_hi, 
#     n_tested_lo, n_tested_hi, 
#     n_positive_lo, 
#     1e-4 
# )
# How you access the returned variables
#m_aggr_prev().n_positive
# Prior Predicitve simulation for prev_gp_aggr func
#m_aggr_prev_pr = sample(m_aggr_prev, Prior(), 10)

#lp = @. b0 + gp_aggr 
positive_hi_preds = []
for s in 1:10
    f_latent_sample = [m_aggr_prev_pr["f_latent[$i]"][s] for i in 1:2618]
    M_hi_sample = M_g(pol_pt_hi, f_latent_sample)
    b0_sample = m_aggr_prev_pr["b0"][s]
    lp_sample = @. b0_sample + M_hi_sample # (n_regions_hi,)
    n_positive_hi_sample = rand(product_distribution(BinomialLogit.(n_tested_hi, lp_sample)))
    @show n_positive_hi_sample
    push!(positive_hi_preds, n_positive_hi_sample)
end

mean(positive_hi_preds, dims = 1) # Mean of the predictions


# Testing Kernel Functions 
fx, f_latent, gp_aggr_lo, gp_aggr_hi = gp_aggrv2(x, pol_pt_lo, pol_pt_hi, 1e-4, SqExponentialKernel())
# Posterior Prediction
#m_aggr_prev_pos = sample(m_aggr_prev, NUTS(), 1)
k = SEKernel()
f = GP(k)
x_svec = [SVector{2,}(x[i,:]) for i in 1:size(x,1)]
f_latent = f(x_svec, 1e-4)
rand(f_latent)

#end