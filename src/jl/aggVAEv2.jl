# ---------------------------------------------------------------------------- #
#                                   avgAggVAE                                   #
# We will be training VAE to reconstruct **averaged aggregated GP** 
# The reason why we are avergaging aggregates is to ensure not a lot 
# latent prevalence values dont stick to 0's and 1's after applying logisitic 

# The VAE is used to speec up MCMC as just having GP within a turing model is 
# extremly slow.
# ---------------------------------------------------------------------------- #
import Pkg 
Pkg.activate("jl_envs/lo2hi-v1.11")
using GeoDataFrames

using Turing
using StaticArrays: SVector
using AbstractGPs: GP, SEKernel, with_lengthscale,ScaleTransform

using Lux 
using Optimisers: AdamW
using Mooncake: AutoMooncake
using Zygote: AutoZygote
using MLUtils:DataLoader
using Random

using Printf: @printf
using JLD2
import JSON
using ArgParse
using CairoMakie

set_theme!(theme_dark())

include("utils.jl")
(using .Utils: M_g_mean, vae_encoder, vae_decoder, VAE, loss_function)

# ----------------------------- Argparse Setting ----------------------------- #
s = ArgParseSettings()
@add_arg_table s begin 
    "--fly"
    help = "generate GP's on fly"
    action = :store_true
    "--epochs"
    help = "Number of epochs"
    arg_type = Int 
    default = 30
    "--n_samples" 
    help = "number of gp samples to generate"
    arg_type = Int 
    default = 5120
    "--b_size"
    help = "Batch Size"
    arg_type = Int
    default = 256
    "--valid_size"
    help = "Percentage of validation data in comparison to train,i.e 1, train_size = valid_size"
    arg_type = Float64
    default = 1.0
end 

args = parse_args(s)

# -------------------------- Datasets & DataLoaders -------------------------- #

@doc """ 
Creates GP samples 
Inputs 
    - df: county level dataframe containing lat/lon 
    - M: Matrix containing 1's or 0's for points in region
    - n_samples : number of gp samples 
    - jitter : jitter to be added to GP 
Outputs
    - f_x : GP's (3117, n_samples)
"""
function get_aggmean_gp(
    df::DataFrame,
    M::Matrix{Int64};
    n_samples::Int =100,
    jitter::Float64 = 1e-4
    )::Matrix{Float32}

    kernel_length = InverseGamma(3,3)
    kernel_variance = truncated(Normal(0,0.05), lower = 0)
    x = df[!, [:centroid_x,:centroid_y]] |> Matrix #(3117,2)
    k = with_lengthscale(rand(kernel_variance) * SEKernel(), rand(kernel_length))
    f = GP(k)
    x_svec = [SVector{2,}(x[i,:]) for i in 1:size(x,1)]
    f_latent = f(x_svec, jitter)
    f_x = rand(f_latent, n_samples) .|> Float32 
    fμ_agg = M_g_mean(M,f_x)
    return fμ_agg
end

@doc """
Training Loop for fixed Dataset 
Inputs 
    - rng : Random Generator (i.e Xoshiro())
    - model : VAE model 
    - opt : Optimizer 
    - trainloader : train dataloader 
    - validloader : valid dataloader 
    - epochs : number of epochs to train 
"""->
function train_fxd(rng, model, opt, trainloader, validloader;epochs = 25)
    Random.seed!(rng)
    ps, st = Lux.setup(rng, model)
    train_state = Training.TrainState(model, ps, st, opt)
    losses = Dict(:train => [], :valid => [])
    for epoch in 1:epochs
        running_train_loss = 0.0f0
        running_valid_loss = 0.0f0
        running_train_samples = 0
        running_valid_samples = 0 
        for (i, (X,)) in enumerate(trainloader)
            (_, loss, _, train_state) = Training.single_train_step!(
                AutoZygote(),
                loss_function,
                X,
                train_state;
                return_gradients = Val(false)
            )
            running_train_loss += loss
            running_train_samples += size(X,2) 
        end
        for (i,(X,)) in enumerate(validloader)
            loss, _ = loss_function(
                train_state.model, 
                train_state.parameters,
                train_state.states,
                X
            )
            running_valid_loss += loss 
            running_valid_samples += size(X,2)
        end
        @printf "Epoch : %d Train Loss : %.3f Valid Loss : %.3f \n" epoch running_train_loss/running_train_samples running_valid_loss/running_valid_samples
        push!(losses[:train], running_train_loss/running_train_samples)
        push!(losses[:valid], running_valid_loss/running_valid_samples)
    end
    return train_state, losses
end

@doc """
Trains models with GP's generated on fly. Every epoch the model will see 
a new set of GP's. Technically you shouldnt need a validation set. 
Inputs 
    - rng : Random Generator 
    - model : VAE model 
    - opt : optimizer 
    - df_grid : dataframe containing lat/lon points at county level 
    - get_cat_gps : A function from `training_loop` function that concatenates lo and hi gps 
    - n_samples : number of samples 
    - epochs : number of epochs 
    - valid_size : validation size in comparison to train size (0-1)
    - jitter : jitter 
"""
function train_fly(
    rng, 
    model, 
    opt, 
    df_grid::DataFrame,
    M_lo::Matrix{Int64},
    M_hi::Matrix{Int64},
    get_cat_gps::Function #This functions is defined in training_loop and returns concatenated hi and lo gp's
    ; 
    n_samples::Int = args.n_samples, 
    epochs::Int = args.epochs, 
    valid_size::Float64 = args.valid_size,
    jitter::Float64 = 1e-4
    )
    Random.seed!(rng)
    ps,st = Lux.setup(rng, model)
    train_state = Training.TrainState(model, ps, st, opt)
    losses = Dict(:train => [], :valid => [])
    for epoch in 1:epochs 
        X_train = get_cat_gps(df_grid, M_lo, M_hi, n_samples = n_samples, valid_size = nothing,  jitter = jitter)
        X_valid = get_cat_gps(df_grid, M_lo, M_hi, n_samples = n_samples, valid_size = valid_size, jitter = jitter)
        (_,train_loss,_,train_state) = Training.single_train_step!(
            AutoZygote(),
            loss_function, 
            X_train, 
            train_state;
            return_gradients = Val(false)
        )
        valid_loss, _ = loss_function(
            train_state.model,
            train_state.parameters,
            train_state.states, 
            X_valid
        )
        push!(losses[:train], train_loss/n_samples)
        push!(losses[:valid], valid_loss/Int(valid_size * n_samples))
        @printf "Epoch : %d Train Loss : %.3f Valid Loss : %.3f \n" epoch train_loss/n_samples valid_loss/Int(valid_size * n_samples)
    end
    return train_state, losses
end

@doc """
    Train VAE for n number of epochs. 
    Inputs 
        - Model : VAE model 
        - opt : Optimizer 
        - df_grid : Dataframe containing lat/lon points at county level 
        - M_lo/M_hi : Martrix containing 1's for points that fall on regions else 0 
        - get_gp : Function that produces GP (Function to be places as it is without arguments)
        - epochs : Number of epochs 
        - n_samples : Number of samples 
        - batchsize : Number of batches (only for fixed dataloader)
        - valid_size : validation data size in relation to training (float value between 0-1)
        - fly : Boolean, if true trains model generating GP's on fly, else  fixed set of GP's using a dataloader
"""->
function training_loop(
    model,
    opt,
    df_grid::DataFrame, 
    M_lo::Matrix{Int64},
    M_hi::Matrix{Int64},
    get_gp::Function
    ;
    epochs::Int = 25,
    n_samples::Int = 100,
    batchsize::Union{Int, Nothing} = nothing,
    valid_size::Float64 = 1.0,
    fly::Bool = true,
    jitter::Float64 = 1e-4,
)
    function get_cat_gps(df_grid,M_lo,M_hi; n_samples, valid_size, jitter)
        if valid_size isa Nothing
            gp_aggmean_lo = get_gp(df_grid,M_lo;n_samples = n_samples, jitter = jitter) #(9,*)
            gp_aggmean_hi = get_gp(df_grid,M_hi;n_samples = n_samples, jitter = jitter) #(49,*)
        else 
            gp_aggmean_lo = get_gp(df_grid,M_lo;n_samples = Int(n_samples * valid_size), jitter = jitter) #(9,*)
            gp_aggmean_hi = get_gp(df_grid,M_hi;n_samples = Int(n_samples * valid_size), jitter = jitter) #(49,*)
        end
        return vcat(gp_aggmean_lo, gp_aggmean_hi) #(58,*)
    end

    if fly 
        println("Training model by generating GP's on fly")
        train_state, losses = train_fly(
            Xoshiro(), 
            model, 
            opt, 
            df_grid, 
            M_lo, 
            M_hi, 
            get_cat_gps; 
            n_samples = n_samples, 
            epochs = epochs, 
            valid_size = valid_size, 
            jitter = jitter
        )
    else 
        println("Training model using fixed datloader")
        gp_aggmean_train = get_cat_gps(df_grid, M_lo, M_hi,n_samples = n_samples, valid_size = nothing, jitter = jitter) #(58,*)
        gp_aggmean_valid = get_cat_gps(df_grid, M_lo, M_hi,n_samples = n_samples, valid_size = valid_size, jitter = jitter) #(58,*')
        trainloader = DataLoader((gp_aggmean_train,), batchsize = batchsize)
        validloader = DataLoader((gp_aggmean_valid,), batchsize = batchsize)
        train_state, losses = train_fxd(
            Random.Xoshiro(),
            model,
            opt,
            trainloader,
            validloader,
            epochs = epochs
        )
    end
    return train_state, losses
end

# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- # 

# --------------------------------- Load Data -------------------------------- #
begin 
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

    # Load Grid : Also contains population at county level
    county_fold = "data/processed/v3/county_grid_v3jl"
    county_files = filter(x -> endswith(x, ".shp"), readdir(county_fold))
    county_path = joinpath(county_fold, first(county_files))
    df_county = GeoDataFrames.read(county_path)

    pol_pts = JLD2.load("data/processed/v3/pol_pts.jld2")
    pol_pts_lo = pol_pts["pol_pts_lo"]
    pol_pts_hi = pol_pts["pol_pts_hi"]

    pt_which_pol = JLD2.load("data/processed/v3/pt_which_pol.jld2")
    pt_which_pol_lo = pt_which_pol["pt_which_pol_lo"]
    pt_which_pol_hi = pt_which_pol["pt_which_pol_hi"]
end

# ----------------------- Create Datasets & DataLoaders ---------------------- #

in_dim = 58 ; h_dim = 50 ; z_dim = 40
model = VAE(Xoshiro(), in_dim, h_dim, z_dim)
opt = AdamW(;eta = 1.0e-3, lambda = 1.0e-5)

epochs = args["epochs"]
n_samples = args["n_samples"]
batch_size = args["b_size"]
valid_size = args["valid_size"]

train_state, losses = training_loop(
    model,
    opt,
    df_county, 
    pol_pts_lo,
    pol_pts_hi,
    get_aggmean_gp;
    epochs = epochs,
    n_samples = n_samples,
    batchsize = batch_size,
    valid_size = valid_size,
    fly = args["fly"],
    jitter = 1e-4, 
)

# --------------------------- Save Model Parameters -------------------------- #
trained_ps = train_state.parameters 
trained_st = train_state.states 
flyorfxd = args["fly"] ? "fly" : "fxd"
save_dir = "model_runs/jl/aggGpVae_ep$(epochs)_smp$(n_samples)_$(flyorfxd)"
if !isdir(save_dir)
    mkpath(save_dir)
end

# Save Model Parameters 
model_params = Dict(
    "input_dim" => in_dim,
    "hidden_dim" => h_dim,
    "z_dim" => z_dim,
    "train_method" => flyorfxd,
    "epoch" => epochs,
    "n_samples" => n_samples,
    "batch_size" => args["fly"] ? nothing : batch_size
)

# Write model parameters to JSON file 
open(joinpath(save_dir, "model_params.json"), "w") do f
    JSON.print(f, model_params)
end 

# Save VAE state and params 
@save joinpath(save_dir, "params_states") {compress = true} trained_ps trained_st 

# ---------------------------------- Losses ---------------------------------- #
fig = Figure(); ax = Axis(fig[1,1], title = "losses")
lines!(1:length(losses[:train]), losses[:train], label = "train")
lines!(1:length(losses[:valid]), losses[:valid], label = "valid")
fig[1,2] = Legend(fig, ax, "losses")
CairoMakie.save(joinpath(save_dir, "losses.png"), fig)