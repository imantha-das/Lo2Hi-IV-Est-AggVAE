begin 
    import Pkg
    Pkg.activate("jl_env/lo2hi")
    using DataFrames 
    using GeoDataFrames 
    import CSV

    using Lux
    using Lux.Experimental: @compact
    using LuxCore
    using MLUtils: rand_like
    using Random
    using Enzyme
    using Zygote
    using Printf
    using Optimisers
    using ConcreteStructs


    using StaticArrays: SVector
    using Turing
    using Turing:logistic
    using AbstractGPs: GP, SEKernel, with_lengthscale, Kernel, SqExponentialKernel, GibbsKernel
end

begin 
    include("utils.jl")
    (using .Utils: compute_pts_in_polys_v3, M_g, plot_gp_aggr)
end

# --------------------------------- GIS Data --------------------------------- #
begin 
    # Load Grid : Also contains population at county level
    county_fold = "data/processed/v3/county_grid"
    county_files = filter(x -> endswith(x, ".shp"), readdir(county_fold))
    county_path = joinpath(county_fold, first(county_files))
    df_county = GeoDataFrames.read(county_path)

    # Load Census : Low Resolution data
    census_fold = "data/processed/v3/census"
    census_files = filter(x -> endswith(x,".shp"), readdir(census_fold)) 
    census_path = joinpath(census_fold, first(census_files))
    df_census = GeoDataFrames.read(census_path)

    # Load States : High resolution data 
    state_fold = "data/processed/v3/states"
    state_files = filter(x -> endswith(x, ".shp"), readdir(state_fold))
    state_path = joinpath(state_fold, state_files[1])
    df_state = GeoDataFrames.read(state_path)

    # Lat/Lon Points 
    coords = zip(df_county.centroid_x, df_county.centroid_y) |> collect

    # Points in Polygons 
    pol_pts_lo, pt_which_pol_lo = compute_pts_in_polys_v3(coords, df_census)
    @assert size(pol_pts_lo)[1] == length(unique(df_census.region)) 
    pol_pts_hi, pt_which_pol_hi = compute_pts_in_polys_v3(coords, df_state)
    @assert size(pol_pts_hi)[1] == length(unique(df_state.region)) 

    df_county[!,:poly_lo] = pt_which_pol_lo;
    df_county[!,:poly_hi] = pt_which_pol_hi;

    # There are a few points that dont fall on either census regions or high region 
    df_county_miss = filter([:poly_lo,:poly_hi] => (x1,x2) -> x1 == 0 || x2 == 0, df_county) 
    not_miss_idx = findall((df_county.poly_hi .!= 0) .& (df_county.poly_hi .!= 0))
    pol_pts_lo = pol_pts_lo[:, not_miss_idx]
    pol_pts_hi = pol_pts_hi[:, not_miss_idx]
    pt_which_pol_lo = pt_which_pol_lo[not_miss_idx]
    pt_which_pol_hi = pt_which_pol_hi[not_miss_idx]
    filter!([:poly_lo,:poly_hi] => (x1,x2) -> x1 != 0 || x2 != 0, df_county) 
end

# ------------------------------ Population Data ----------------------------- #
begin 
    # We might not use these population data even though they are 
    # loaded. The reason is we will be using county populaiton to 
    # normalize the GP. Regardless we load it for the moment 
    
    df_census_pop = CSV.read("data/processed/popn/census_popn.csv", DataFrame)
    df_state_pop = CSV.read("data/processed/popn/state_popn.csv", DataFrame)
    rename!(df_census_pop, :division => :region)
    rename!(df_state_pop, :state => :region)
end

# ------------------------------ Influenza Data ------------------------------ #
begin 
    # Load Influenza Data
    infz_data_root = "data/processed/fluview/nrevss_phl_20152024"
    df_infz_census_1524 = CSV.read(joinpath(infz_data_root, "infz_census_20152024.csv"), DataFrame)
    df_infz_state_1524 = CSV.read(joinpath(infz_data_root, "infz_state_20152024.csv"), DataFrame)
    
    # Filter Data Based on Year 
    filter_year = 2018 
    df_infz_census = filter(:year => ==(filter_year), df_infz_census_1524)
    df_infz_state = filter(:year => ==(filter_year), df_infz_state_1524)
    select!(df_infz_census, :region, :year, :test_cases, :tested_pos)
    select!(df_infz_state, :region, :year, :test_cases, :tested_pos)
end

# ---------------------------------- Joining --------------------------------- #
begin 
    # Join Data with GeoDataFrames 
    df_lo_all = innerjoin(df_census, df_infz_census, df_census_pop, on = :region)
    df_hi_all = innerjoin(df_state, df_infz_state, df_state_pop, on = :region)
    # There are multiple geometries corresponding the same region 
    # in census regions. This is an issue, so we need to find 
    # unique regions 
    df_lo = unique(select(df_lo_all, :region, :year, :test_cases, :tested_pos, :tot_popn))
    df_hi = unique(select(df_hi_all, :region, :year, :test_cases, :tested_pos, :tot_popn))
end

# ---------------------------------- Test GP --------------------------------- #
let 
    x = df_county[!, [:centroid_x,:centroid_y]] |> Matrix #(3117,2)
    k = SEKernel()
    f = GP(k)
    x_svec = [SVector{2,}(x[i,:]) for i in 1:size(x,1)]
    f_latent = f(x_svec, 1e-4)
    n_samples = 100
    f_reals = rand(f_latent, n_samples)
    μp = rand(Normal(0,1),n_samples)
    p_logits = reshape(μp, (1,100)) .+ f_reals #(3117,100)
    p = logistic.(p_logits)

    # Population weighting 
    x_pop = df_county[!, :pop2021]
    x_pop_w = x_pop ./ sum(x_pop)
    p_w = p .* x_pop_w
    p_lo = M_g(pol_pts_lo, p_w) 
    p_hi = M_g(pol_pts_hi, p_w) 
    
    fig1 = plot_gp_aggr(vcat(p_lo,p_hi), vcat(df_lo.region, df_hi.region), title = "Prevalence prior for gp aggregated")
    fig1
end 

# ---------------------------------------------------------------------------- #
#                                      VAE                                     #
# ---------------------------------------------------------------------------- #

# ----------------------------- Encoder & Decoder ---------------------------- #
# function vae_encoder(rng, inp_dims::Int,h_dims::Int, z_dims::Int)
#     return @compact(;
#         fc1 = Dense(inp_dims,h_dims), #(inp_dims, *) -> (h_dims, *)
#         bn1 = BatchNorm(h_dims), #(h_dims, *) -> (h_dims, *)
#         fc_mu = Dense(h_dims, z_dims), #(h_dims, *) -> (z_dims, *)
#         fc_logvar = Dense(h_dims, z_dims), #(h_dims, *) -> (z_dims, *)
#         rng
#     ) do x 
#         h = elu.(bn1(fc1(x))) #(h_dims, *)
#         μ = fc_mu(h) #(z_dims, *)
#         logσ² = fc_logvar(h) #(z_dims, *) 
#         # Clamp log variance for numerical stability 
#         T = eltype(logσ²)
#         logσ² = clamp.(logσ², -T(20.0f0), T(10.0f0)) #(z_dims, *)
#         σ = exp.(logσ² .* T(0.5)) #(z_dims, *)
#         # Generate a tensor of random values from a normal distribution 
#         ϵ = randn_like(Lux.replicate(rng), σ)
#         # Reparameterization trick
#         z = μ .+ σ .* ϵ
#         @return z, μ, logσ² 
#     end
# end

# function vae_decoder(z_dim, h_dims, out_dims)
#     return @compact(;
#         fc1 = Dense(z_dim, h_dims), #(z_dim, *) -> (h_dims, *)
#         bn1 = BatchNorm(h_dims), #(h_dims, *) -> (h_dims, *)
#         fc2 = Dense(h_dims, out_dims), #(h_dims, *) -> (out_dims, *)
#     ) do Z 
#         h = elu.(bn1(fc1(Z))) #(h_dims, *)
#         gp_recon = fc2(h)
#         @return gp_recon
#     end
# end

function vae_encoder_gp_grid(rng, inp_dims::Int,h_dims::Int, z_dims::Int)
    return @compact(;
        fc1 = Dense(inp_dims, h_dims), #(3117, *) -> (1558, *)
        bn1 = BatchNorm(h_dims),
        fc2 = Dense(h_dims, h_dims ÷ 2), # (1558, *) -> (779, *)
        bn2 = BatchNorm(h_dims ÷ 2),
        fc_mu = Dense(h_dims ÷ 2, z_dims), #(779, *) -> (389, *)
        fc_logvar =  Dense(h_dims ÷ 2, z_dims) #(779, *) -> (389, *)
    
    ) do x 
    out = elu.(bn1(fc1(x))) #(3117, *) -> (1558, *)
    out = elu.(bn2(fc2(out))) #(1558, *) -> (779, *)
    μ = fc_mu(out) #(779, *) -> (389, *)
    logσ² = fc_logvar(out)
    # Clamp log variance for numerical stability
    T = eltype(logσ²)
    logσ² = clamp.(logσ², -T(20.0f0), T(10.0f0))
    σ = exp.(logσ² .* T(0.5))
    # Generate a tensor of random value from a normal dist 
    ϵ = rand_like(Lux.replicate(rng), σ)
    # Reparameterization trick 
    z = μ .+ σ .* ϵ
    return z, μ, logσ²
end
end 

function vae_decoder_gp_grid(z_dim, h_dims, out_dims)
    return @compact(;
    fc1 = Dense(z_dim, h_dims * 2), #(389, *) -> (779, *)
    bn1 = BatchNorm(h_dims * 2),
    fc2 = Dense(h_dims * 2, h_dims), #(779, *) -> (1558, *)
    bn2 = BatchNorm(h_dims),
    fc3 = Dense(h_dims, out_dims) #(1558, *) -> (3117, *)
    ) do Z 
        out = elu.(bn1(fc1(Z)))  #(389, *) -> (779, *)
        out = elu.(bn2(fc2(out))) #(779, *) -> (1558, *)
        gp_recon = fc3(out) #(1558, *) -> (3117, *)
        @return gp_recon
    end
end

# Test code 
let 
    x = df_county[!, [:centroid_x,:centroid_y]] |> Matrix #(3117,2)
    k = SEKernel()
    f = GP(k)
    x_svec = [SVector{2,}(x[i,:]) for i in 1:size(x,1)]
    f_latent = f(x_svec, 1e-4)
    n_samples = 100
    f_reals = rand(f_latent, n_samples) .|> Float32
    vae_enc = vae_encoder_gp_grid(Random.default_rng(), size(f_reals,1),1558,389)
    ps, st = Lux.setup(Random.default_rng(), vae_enc)
    (z,μ, logσ²), newstate = vae_enc(f_reals, ps, st)
    
    vae_dec = vae_decoder_gp_grid(389,1558,size(f_reals,1))
    ps_dec, st_dec = Lux.setup(Random.default_rng(), vae_dec)
    f_real_recon, newstate_dec = vae_dec(z, ps_dec, st_dec)
    f_real_recon
end

# --------------------------------- VAE Class -------------------------------- #
@concrete struct VAE <: AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder <: AbstractLuxLayer
    decoder <: AbstractLuxLayer 
end 

# function VAE(rng, h_dims, z_dims, inp_dims, out_dims)
#     decoder = vae_decoder(z_dims, h_dims, out_dims) 
#     encoder = vae_encoder(rng, inp_dims, h_dims, z_dims)
#     return VAE(encoder, decoder) 
# end 

function VAE(rng, h_dims, z_dims, inp_dims, out_dims)
    decoder = vae_decoder_gp_grid(z_dims, h_dims, out_dims)
    encoder = vae_encoder_gp_grid(rng, inp_dims, h_dims, z_dims)
    return VAE(encoder, decoder)
end

# Forward function
function (vae::VAE)(x, ps, st)
    (z, μ, logσ²), st_enc = vae.encoder(x, ps.encoder, st.encoder)
    x_rec, st_dec = vae.decoder(z, ps.decoder, st.decoder)
    return (x_rec, μ, logσ²), (;encoder = st_enc, decoder = st_dec)
end

function encode(vae::VAE, x, ps, st)
    (z, _,_), st_enc = vae.encoder(x, ps.encoder, st.encoder)
    return z, (;encoder = st_enc, st.decoder) #you need to return decoder to update sttes
end

function decode(vae::VAE, z, ps, st)
    gp_rec, st_dec = vae.decoder(z, ps.decoder, st.decoder)
    return gp_rec, (;decoder = st_dec, st.encoder) #you need to return encoder to update its state
end

# --------------------------- Dataset & Dataloaders -------------------------- #
# Fixed Dataset
@doc """
We are reconstructing a noisy gp at grid level, 
shape : (3107,n_samples)
"""->
function make_fixed_dataset_at_grid_level(f_latent;num_batches,batch_size)
    gp_grid = rand(f_latent, batch_size * num_batches) #(3107, num_samples * num_batches)
    return gp_grid 
end

@doc """ 
We are reconstructing aggreagted prevalence values (These are structured) 
shape : (58, n_samples)
"""->
function make_fixed_dataset_at_agg_level(f_latent, M_lo, M_hi, x_pop;num_batches, batch_size)
    n_samples = batch_size * num_batches
    f_reals = rand(f_latent, n_samples) # (3117, *)

    μp = rand(Normal(0,1),n_samples) #(3117, *)
    p_logits = reshape(μp, (1,n_samples)) .+ f_reals #(3117,*)
    p = logistic.(p_logits) #(3117, *)

    # Population weighting 
    x_pop_w = x_pop ./ sum(x_pop)
    p_w = p .* x_pop_w
    p_lo = M_g(M_lo, p_w) 
    p_hi = M_g(M_hi, p_w) 

    #gp_aggr_lo = M_g(M_lo, rand(f_latent, batch_size * num_batches))
    #gp_aggr_hi = M_g(M_hi, rand(f_latent, batch_size * num_batches))
    return vcat(p_lo, p_hi) # (58, num_samples * num_batches)
end 


struct FixedDataLoader
    data::Array #(58,6400)
    batch_size::Int # 64
end 

Base.iterate(loader::FixedDataLoader, state = 1) = 
    # when state > 6400 ? nothing <- do not iterate further
    # if the state = 1 we will get [:,1:64] and state = 65 (second entry of tuple)
    # so next state = 65, we will get [:, 65:128] and state = 129 
    state > size(loader.data, 2) ? nothing : 
    (loader.data[:, state:state+loader.batch_size-1], state + loader.batch_size)


begin 
    batch_size = 128
    num_batches = 100
    valid_size = 0.3
    train_idxs = 1:floor(Int, (1-valid_size) * num_batches * batch_size)
    valid_idxs = (floor(Int, (1-valid_size) * num_batches * batch_size) + 1):num_batches * batch_size
    
    x = df_county[!, [:centroid_x,:centroid_y]] |> Matrix #(3117,2)
    k = SEKernel()
    f = GP(k)
    x_svec = [SVector{2,}(x[i,:]) for i in 1:size(x,1)]
    f_latent = f(x_svec, 1e-4)

    x_pop = df_county[!, :pop2021]
end 

let
    # Test dataset funcs 
    gp_grid_test = make_fixed_dataset_at_grid_level(
        f_latent; 
        num_batches = 100,batch_size = 32
    )
    
    p_test = make_fixed_dataset_at_agg_level(
        f_latent, 
        pol_pts_lo, pol_pts_hi,
        x_pop,
        num_batches = 100, batch_size = 32
    )
    @show size(gp_grid_test) size(p_test)
end

begin  
    gp_grid_fixed = make_fixed_dataset_at_grid_level(
        f_latent;
        batch_size = batch_size,
        num_batches = num_batches
    ); #(3117, *)
end

trainloader = FixedDataLoader(gp_grid_fixed[:,train_idxs] .|> Float32, batch_size)
validloader = FixedDataLoader(gp_grid_fixed[:,valid_idxs] .|> Float32, batch_size)
for batch in trainloader
    @show size(batch)
    @show typeof(batch)
    break
end

# ------------------------------- Loss Function ------------------------------ #
function loss_function(model, ps, st, X)
    (X_recon, μ, logσ²), st = model(X, ps, st)
    reconstruction_loss = MSELoss(agg = sum)(X_recon, X)
    kldiv_loss = -sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²)) / 2
    loss = reconstruction_loss + kldiv_loss 
    return loss, st, (;X_recon, μ, logσ², reconstruction_loss, kldiv_loss)
end

let 
    #test loss function
    X = Float32.(rand(f_latent, 5)) #(3117,*)
    h_dim = 1558 
    z_dim = 389
    vae = VAE(Random.default_rng(), h_dim, z_dim, size(X,1), size(X,1))
    ps, st = Lux.setup(Random.default_rng(), vae)
    loss_function(vae, ps, st, X)
end

# ------------------------------ Training Loop ------------------------------- #
seed = 0
h_dims = 1558
z_dims = 389
inp_dims = out_dims = size(rand(f_latent),1)
learning_rate = 1.0e-3
weight_decay = 1.0e-5
epochs = 50

rng = Xoshiro() 
Random.seed!(rng, seed)
#vae = VAE(rng, h_dims, z_dims, inp_dims, out_dims)
vae = VAE(rng, h_dims, z_dims, inp_dims, out_dims)
ps, st = Lux.setup(rng, vae)

opt = AdamW(; eta = learning_rate, lambda = weight_decay)
train_state = Training.TrainState(vae, ps, st, opt)
@printf "Total Trainable Parameters: %0.4f M\n" (Lux.parameterlength(ps) / 1.0e-6)

for epoch in 1:epochs
    total_loss= 0.0f0 
    total_valid_loss = 0.0f0
    total_samples = 0 
    total_valid_samples = 0

    start = time()
    for (i,X) in enumerate(trainloader)
        (_, loss, _, train_state) = Training.single_train_step!(
            AutoZygote(), loss_function, X, train_state; return_gradients = Val(false)
        )
        total_loss += loss 
        total_samples += size(X, ndims(X))
        throughput = total_samples / (time() - start)
        #@printf "Epoch %d, Iter %d, Loss: %.7f, Throughput: %.6f im/s\n" epoch i loss throughput
    end

    for (i,X) in enumerate(validloader)
        loss, _ = loss_function(train_state.model, train_state.parameters, train_state.states, X)
        total_valid_loss += loss
        total_valid_samples += size(X, ndims(X))

    end
    
    @printf "Epoch: %d Loss: %.3f Valid Loss: %.3f \n" epoch total_loss / total_samples total_valid_loss / total_valid_samples

end

# ------------------------------ Reconstruct GP ------------------------------ #
tps = train_state.parameters
tst = train_state.states 

using CairoMakie
set_theme!(theme_light())
let 
    z_dim = 389
    n_samples = 1
    z = randn(Float32, z_dim, n_samples)
    tst_eval = Lux.testmode(tst)
    f_approx, st_dec = decode(vae, z, tps, tst_eval)
    fig = Figure()
    ax = Axis(fig[1,1])
    for i = 1:size(f_approx,2)
        lines!(ax, 1:size(f_approx,1), f_approx[:,i], color = (:black, 0.3))
    end
    μp = rand(Normal(0,1),n_samples)
    p_logits = reshape(μp, (1,n_samples)) .+ f_approx #(3117,100)
    p = logistic.(p_logits)

    # Population weighting 
    x_pop = df_county[!, :pop2021]
    x_pop_w = x_pop ./ sum(x_pop)
    p_w = p .* x_pop_w
    p_lo = M_g(pol_pts_lo, p_w) 
    p_hi = M_g(pol_pts_hi, p_w) 
    
    #fig1 = plot_gp_aggr(vcat(p_lo,p_hi), vcat(df_lo.region, df_hi.region), title = "Prevalence prior for vae-gp aggregated")
end

#! ---------------------------------------------------------------------------- #
#!                                 TO BE DELETED                                #
#! ---------------------------------------------------------------------------- #

@model function prev_vaegp_aggr_betabin_v1(
    x_pop::Vector{Float64},
    M_lo::Matrix{Int64},
    M_hi::Matrix{Int64},
    n_tested_lo::Vector{Int64},
    n_tested_hi::Vector{Int64},
    n_positive_lo::Vector{Int64};
    jitter::Float64 = 1e-4,
    z_dim::Int = 389
)
    # GP realization : Construct through VAE 
    z = randn(Float32, z_dim, 1) #(z_dim, 1)
    f_approx, _ = decode(vae, z, tps, tst_eval) #(n_pts, 1)

    # Prevalence model
    μp ~ Normal(0, 1)             # mean prevalence
    logits = @. μp + f_approx
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


@model function prev_vaegp_aggr_betabin_v2(
    x_pop::Vector{Float64},
    M_lo::Matrix{Int64},
    M_hi::Matrix{Int64},
    n_tested_lo::Vector{Int64},
    n_tested_hi::Vector{Int64},
    n_positive_lo::Vector{Int64};
    jitter::Float64 = 1e-4,
    z_dim::Int = 389,
)
    # GP realization : Construct through VAE 
    z = randn(Float32, z_dim, 1) #(z_dim, 1)
    f_approx, _ = decode(vae, z, tps, tst_eval) #(n_pts, 1)

    # Prevalence model
    μp ~ Normal(0, 1)             # mean prevalence
    logits = @. μp + f_approx
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

tst_eval = Lux.testmode(tst)
x_pop = df_county[!, :pop2021]
x_pop_w = x_pop ./ sum(x_pop)
n_tested_lo = Vector{Int64}(df_lo[!,:test_cases])
n_tested_hi = Vector{Int64}(df_hi[!,:test_cases])
n_positive_lo = Vector{Int64}(df_lo[!,:tested_pos])
n_positive_hi = Vector{Int64}(df_hi[!, :tested_pos])
#f_approx, st_dec = decode(vae, z, tps, tst_eval)
model = prev_vaegp_aggr_betabin_v1(
    x_pop_w,
    pol_pts_lo,
    pol_pts_hi,
    n_tested_lo,
    n_tested_hi,
    n_positive_lo
)

#posterior_samples = sample(model, NUTS(5,0.65), 100)
posterior_samples = sample(model, NUTS(), 1000)
pos_df = posterior_samples |> DataFrame

pos_df[!,"n_positive[5]"] |> unique