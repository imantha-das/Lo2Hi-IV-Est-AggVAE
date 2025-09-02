# ============================= Variational Autoencoder ============================ #
# Implement AggVAE approach in Lux

# ------------------------------ Load Packages ------------------------------- #
using Lux
using Lux.Experimental: @compact
using LuxCore
using MLUtils: randn_like
using Random
using ConcreteStructs
using Optimisers
using Printf
using Enzyme
using Zygote

using JLD2: @save, @load, load
using JLD2

using .Utils: dist_euclid, exp_sq_kernel, M_g
using NPZ: npzread
using AbstractGPs: GP, SEKernel, with_lengthscale, Kernel, SqExponentialKernel, GibbsKernel

# ----------------------------------- Data ----------------------------------- #
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
    n_tested_lo = Vector{Int64}(df_lo[:, "tot_specs"]);
    n_tested_hi = Vector{Int64}(df_hi[:, "tot_specs"]);
    # Get number tested positive - This is "y" ~ Binomial(n,p)
    n_positive_lo =Vector{Int64}(df_lo[:, "tot_cases"]);
    # Note "p" is prevelence which we model as linear function with Random effects GP
    n_positive_hi = Vector{Int64}(df_hi[:, "tot_cases"]);
end

# GP Kernel
k = SEKernel()
f = GP(k)
x_svec = [SVector{2,}(x[i,:]) for i in 1:size(x,1)]
f_latent = f(x_svec, 1e-4)
# You can extract a realization of the GP using rand
rand(f_latent)

lo_regions = skipmissing(df_lo.area) |> collect
hi_regions = skipmissing(df_hi.area) |> collect

gp_aggr_lo = M_g(pol_pt_lo, rand(f_latent,100))
gp_aggr_hi = M_g(pol_pt_hi, rand(f_latent,100))
gp_aggr = vcat(gp_aggr_lo, gp_aggr_hi) #(58,100)

# ---------------------------------- Encoder --------------------------------- #
function vae_encoder(rng, inp_dims::Int,h_dims::Int, z_dims::Int)
    return @compact(;
        fc1 = Dense(inp_dims,h_dims), #(inp_dims, *) -> (h_dims, *)
        bn1 = BatchNorm(h_dims), #(h_dims, *) -> (h_dims, *)
        fc_mu = Dense(h_dims, z_dims), #(h_dims, *) -> (z_dims, *)
        fc_logvar = Dense(h_dims, z_dims), #(h_dims, *) -> (z_dims, *)
        rng
    ) do x 
        h = elu.(bn1(fc1(x))) #(h_dims, *)
        μ = fc_mu(h) #(z_dims, *)
        logσ² = fc_logvar(h) #(z_dims, *) 
        # Clamp log variance for numerical stability 
        T = eltype(logσ²)
        logσ² = clamp.(logσ², -T(20.0f0), T(10.0f0)) #(z_dims, *)
        σ = exp.(logσ² .* T(0.5)) #(z_dims, *)
        # Generate a tensor of random values from a normal distribution 
        ϵ = randn_like(Lux.replicate(rng), σ)
        # Reparameterization trick
        z = μ .+ σ .* ϵ
        @return z, μ, logσ² 
    end
end

# ---------------------------------- Decoder --------------------------------- #
function vae_decoder(z_dim, h_dims, out_dims)
    return @compact(;
        fc1 = Dense(z_dim, h_dims), #(z_dim, *) -> (h_dims, *)
        bn1 = BatchNorm(h_dims), #(h_dims, *) -> (h_dims, *)
        fc2 = Dense(h_dims, out_dims), #(h_dims, *) -> (out_dims, *)
    ) do Z 
        h = elu.(bn1(fc1(Z))) #(h_dims, *)
        gp_recon = fc2(h)
        @return gp_recon
    end
end

# test encoder and decoder
vae_enc = vae_encoder(Random.default_rng(), size(gp_aggr,1), 50, 40)
ps, st = Lux.setup(Random.default_rng(),vae_enc)
(z,μ, logσ²), newstate =  vae_enc(Float32.(gp_aggr), ps, st) 

vae_dec = vae_decoder(40, 50, 58)
ps, st = Lux.setup(Random.default_rng(),vae_dec)
gp_recon, newstate = vae_dec(z, ps, st)

# ------------------------------------ VAE ----------------------------------- #
@concrete struct VAE <: AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder <: AbstractLuxLayer
    decoder <: AbstractLuxLayer 
end 

function VAE(rng, h_dims, z_dims, inp_dims, out_dims)
    decoder = vae_decoder(z_dims, h_dims, out_dims) 
    encoder = vae_encoder(rng, inp_dims, h_dims, z_dims)
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

# test vae model
vae = VAE(Random.default_rng(), 50, 40, size(gp_aggr,1), size(gp_aggr,1))
ps, st = Lux.setup(Random.default_rng(), vae)
(x_rec, μ, logσ²), st = vae(Float32.(gp_aggr), ps, st)
z, (st_enc, st_dec) = encode(vae, Float32.(gp_aggr), ps, st)
gp_rec, (st_dec, st_enc) = decode(vae, z, ps, st)

# ------------------------------- Loss function ------------------------------ #
function loss_function(model, ps, st, X)
    (X_recon, μ, logσ²), st = model(X, ps, st)
    reconstruction_loss = MSELoss(agg = sum)(X_recon, X)
    kldiv_loss = -sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²)) / 2
    loss = reconstruction_loss + kldiv_loss 
    return loss, st, (;X_recon, μ, logσ², reconstruction_loss, kldiv_loss)
end

#test loss function
X = Float32.(gp_aggr) #(58,*)
vae = VAE(Random.default_rng(), 50, 40, 58, 58)
ps, st = Lux.setup(Random.default_rng(), vae)
loss_function(vae, ps, st, X)

# --------------------------- Create Fixed Dataset --------------------------- #

# Fixed Data Loader 
function make_fixed_dataset(f_latent, M_lo, M_hi;num_batches, batch_size)
    gp_aggr_lo = M_g(M_lo, rand(f_latent, batch_size * num_batches))
    gp_aggr_hi = M_g(M_hi, rand(f_latent, batch_size * num_batches))
    return vcat(gp_aggr_lo, gp_aggr_hi) # (58, num_samples * num_batches)
end 

batch_size = 128
num_batches = 1000
valid_size = 0.3
train_idxs = 1:floor(Int, (1-valid_size) * num_batches * batch_size)
valid_idxs = (floor(Int, (1-valid_size) * num_batches * batch_size) + 1):num_batches * batch_size


# Create a dataset
gp_aggr_fixed = make_fixed_dataset(
    f_latent,
    pol_pt_lo,
    pol_pt_hi;
    batch_size = batch_size,
    num_batches = num_batches
); #(58, 160)

gp_aggr_fixed_val = make_fixed_dataset(
    f_latent,
    pol_pt_lo,
    pol_pt_hi;
    batch_size = batch_size,
    num_batches = 10
)

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

trainloader = FixedDataLoader(gp_aggr_fixed[:,train_idxs] .|> Float32, batch_size)
validloader = FixedDataLoader(gp_aggr_fixed[:,valid_idxs] .|> Float32, batch_size)
for batch in trainloader
    @show size(batch)
    @show typeof(batch)
    break
end

# -------------------------- On the fly Data Loader -------------------------- #
#todo : To be implemented 

# ------------------------------ Training Loop ------------------------------- #
seed = 0
h_dims = 50
z_dims = 40
imp_dims = out_dims = size(gp_aggr,1)
learning_rate = 1.0e-3
weight_decay = 1.0e-5
epochs = 25

rng = Xoshiro() 
Random.seed!(rng, seed)
vae = VAE(rng, h_dims, z_dims, imp_dims, out_dims)
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

# ------------------------------ Save the model ------------------------------ #
if !isdir("checkpoints_jl")
    mkpath("checkpoints_jl")
end

trained_params = train_state.parameters
trained_states = train_state.states
@save "checkpoints/vae_params_states.jld2" {compress = true} trained_params trained_states

