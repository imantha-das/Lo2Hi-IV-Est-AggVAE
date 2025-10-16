# ---------------------------------------------------------------------------- #
#                                    GP VAE                                    #
# aggVAE : Reconstructs the aggregated GP's 
# gpVAE : Reconstructs GP's (not aggregated)
# The reason behind using VAE's is samplers such as NUTS are 
# extremely slow (in julia's case doesnt even evaluate) with GP's.
# By approximating GP's with VAE's you can get a significant speed up 

# Here, we attempt to reconstruct the GP's themselves instead of the regionwise aggregated ones. 
# This is a much noisy but once we aggragate approximated gp realization (through VAE)
# we end up with structured GP's similar to aggVAE. 
# One reason why this was performed is to perform population weighting at Grid points (county level)
# This way the sum of prevalence ≈ 1.
# ------------------------------------- . ------------------------------------ #

import Pkg
Pkg.activate("jl_envs/lo2hi")
using DataFrames 
using GeoDataFrames

using MLUtils: DataLoader
using Lux
using Lux: Training, AutoZygote
using Zygote
using Random

using Printf 
using Optimisers 

using JLD2 
using StaticArrays: SVector 

using Turing 
using Turing:logistic 
using AbstractGPs: GP, SEKernel, with_lengthscale, Kernel, SqExponentialKernel

include("utils.jl")
(using .Utils: 
compute_pts_in_polys_v3, M_g, plot_gp_aggr,
vae_encoder, vae_decoder, VAE, loss_function
)

import JSON
using ArgParse
using CairoMakie
set_theme!(theme_light())

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
    default = 512
    "--b_size"
    help = "Batch Size"
    arg_type = Int
    default = 64
    "--valid_size"
    help = "Percentage of validation data in comparison to train,i.e 1, train_size = valid_size"
    arg_type = Float64
    default = 1.0
end 

args = parse_args(s)

# --------------------------- Dataset & DataLoaders -------------------------- #

@doc """ 
Creates GP samples 
Inputs 
    - df: county level dataframe containing lat/lon 
    - n_samples : number of gp samples 
    - jitter : jitter to be added to GP 
Outputs
    - f_x : GP's (3117, n_samples)
"""
function make_gp_grid(df::DataFrame,n_samples::Int;jitter::Float64 = 1e-4)::Matrix{Float32}
    x = df[!, [:centroid_x,:centroid_y]] |> Matrix #(3117,2)
    k = SEKernel()
    f = GP(k)
    x_svec = [SVector{2,}(x[i,:]) for i in 1:size(x,1)]
    f_latent = f(x_svec, jitter)
    f_x = rand(f_latent, n_samples) .|> Float32 
    return f_x
end

@doc """
Training Loop for a fixed Dataset 
Inputs
    - rng : Random Generator (instantiated, i.e Xoshiro())
    - model : VAE model 
    - opt : Optimizer 
    - trainloader : train dataloader 
    - validloader : valid dataloader 
    - epochs : number of epochs to train 
Outputs 
    - train_state : Training State object, can be used to access params & states 
    - losses : dictionary containing losses
"""->
function train_fixed(rng,model,opt,trainloader,validloader;epochs = 30)
    Random.seed!(rng)
    ps,st = Lux.setup(rng,model)
    train_state = Training.TrainState(model, ps,st,opt)
    losses = Dict(:train => [], :valid => [])
    for epoch = 1:epochs
        running_train_loss = 0.0f0
        running_valid_loss = 0.0f0 
        running_train_samples = 0
        running_valid_samples = 0 
        start = time()
        for (i,(X,)) in enumerate(trainloader)
            #@show typeof(X[1])
            (_, loss, _, train_state) = Training.single_train_step!(
                AutoZygote(),
                loss_function,
                X,
                train_state;
                return_gradients = Val(false)
            )
            running_train_loss += loss 
            running_train_samples += size(X,2) # batch dim 
            throughput = running_train_samples / (time() - start) 
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
Training Loop for gp constructed on fly
Unlike having a fixed approach, this method will generate gp's on the fly
meaning that during every epoch a new set of GP's will be seen by the model 
Inputs 
    - rng : Random Generator 
    - model : VAE model 
    - opt : Optimizer 
    - n_samples : number of gp's generated 
    - epochs : number of epochs model is trained for 
Outputs 
    - train_state : contains parameters and states of model 
    - losses : dictionary containing losses
"""->
function train_fly(rng, model,opt, df_grid;n_samples = 100, epochs = 30, valid_size = 1)
    Random.seed!(rng)
    ps,st = Lux.setup(rng, model)
    train_state = Training.TrainState(model, ps, st, opt)
    losses = Dict(:train => [], :valid => [])
    for epoch = 1:epochs
        X_train = make_gp_grid(df_grid,n_samples) #(3117,*)
        X_valid = make_gp_grid(df_grid,Int(n_samples*valid_size)) #(3117,*)
        (_, train_loss, _, train_state) = Training.single_train_step!(
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
        push!(losses[:train], train_loss)
        push!(losses[:valid], valid_loss)
        @printf "Epoch : %d Train Loss : %.3f Valid Loss : %.3f \n" epoch train_loss/n_samples valid_loss/Int(valid_size * n_samples)
    end
    return train_state, losses
end

function training_loop(
    model,
    opt,
    epochs::Int,
    df_grid::DataFrame,
    n_samples::Int,
    batchsize::Union{Int,Nothing},
    valid_size::Float64,
    fly::Bool
    )
    if fly
        println("Training models on Gp's generated on fly")
        train_state, losses = train_fly(
            Random.Xoshiro(), 
            model, 
            opt, 
            df_grid; 
            n_samples = n_samples,
            epochs = epochs, 
            valid_size = valid_size
        )
    else
        print("Training model on fixed set of Gp's")
        trainset = make_gp_grid(df_grid, n_samples)
        validset = make_gp_grid(df_grid, Int(n_samples * valid_size))
        trainloader = DataLoader((trainset,), batchsize = batchsize)
        validloader = DataLoader((validset,), batchsize = batchsize)
        train_state, losses = train_fixed(
            Random.Xoshiro(),
            model,
            opt,
            trainloader,
            validloader;
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

# ----------------------------- Model + Training ----------------------------- #

in_dim = 3117 
z_dim =389
model = VAE(Random.Xoshiro(), in_dim, z_dim)
opt = AdamW(;eta = 1.0e-3, lambda = 1.0e-5)
epochs = args["epochs"]
n_samples = args["n_samples"]
batch_size = args["b_size"]
valid_size = args["valid_size"]
fly = args["fly"]
train_state, losses = training_loop(
    model,
    opt,
    epochs,
    df_county,
    n_samples,
    batch_size,
    valid_size,
    fly,
)

# ---------------------------- Saving Model Params --------------------------- #

trained_ps = train_state.parameters 
trained_st = train_state.states
flyorfxd = args["fly"] ? "fly" : "fxd"
save_dir = "model_runs/jl/gpvae_ep$(epochs)_smp$(n_samples)_$(flyorfxd)"
if !isdir(save_dir)
    mkpath(save_dir)
end
# Save Model parameters 
model_params = Dict(
    "input_dim" => in_dim, 
    "z_dim" => z_dim, 
    "h1_dim" => in_dim ÷ 2,
    "h2_dim" => in_dim ÷ 4,
    "dataloader" => flyorfxd,
    "epochs" => epochs,
    "n_samples" => n_samples,
    "batch_size" => args["fly"] ? nothing : batch_size
)

# Write Model params to JSON file 
open(joinpath(save_dir, "model_params.json"), "w") do f 
    JSON.print(f, model_params)
end

# Save VAE State and Params 
@save joinpath(save_dir,"params_states") {compress = true} trained_ps trained_st

# -------------------------------- Plot losses ------------------------------- #
fig = Figure();
ax = Axis(fig[1,1], title = "losses");
lines!(1:length(losses[:train]), losses[:train], color = (:blue), label = "train")
lines!(1:length(losses[:valid]), losses[:valid], color = (:green), label = "valid")
fig[1,2] = Legend(fig, ax, "losses")
CairoMakie.save(joinpath(save_dir,"losses.png"), fig)
