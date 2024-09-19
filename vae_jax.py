import os
import math
import numpy as np 

import jax 
import jax.numpy as jnp 
from jax import random, lax, jit, ops
from jax.example_libraries import stax

import numpyro 
from numpyro.infer import SVI, MCMC, NUTS, init_to_median, Predictive, RenyiELBO 
import numpyro.distributions as dist  

import geopandas as gpd
import plotly.express as px

from termcolor import colored
# ---------------------------- GP Kernel Function ---------------------------- #
def dist_euclid(x, z):
    x = jnp.array(x) 
    z = jnp.array(z)
    if len(x.shape)==1:
        x = x.reshape(x.shape[0], 1)
    if len(z.shape)==1:
        z = x.reshape(x.shape[0], 1)
    n_x, m = x.shape
    n_z, m_z = z.shape
    assert m == m_z
    delta = jnp.zeros((n_x,n_z))
    for d in jnp.arange(m):
        x_d = x[:,d]
        z_d = z[:,d]
        delta += (x_d[:,jnp.newaxis] - z_d)**2
    return jnp.sqrt(delta)


def exp_sq_kernel(x, z, var, length, noise, jitter=1.0e-4):
    dist = dist_euclid(x, z)
    deltaXsq = jnp.power(dist/ length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    k += (noise + jitter) * jnp.eye(x.shape[0])
    return k

# --------------------------- Aggregation functions -------------------------- #
def M_g(M, g):
    '''
    - $M$ is a matrix with binary entries $m_{ij},$ showing whether point $j$ is in polygon $i$
    - $g$ is a vector of GP draws over grid
    - $maltmul(M, g)$ gives a vector of sums over each polygon
    '''
    M = jnp.array(M)
    g = jnp.array(g).T
    return(jnp.matmul(M, g))

# ------------------- Func for Prior Predictive Simulation ------------------- #
def gp_aggr(args):
    x = args["x"]
    gp_kernel = args["gp_kernel"]
    noise = args["noise"]
    
    M1 = args["M1"] #M_old in original code
    M2 = args["M2"] #M_new in original code

    consider_m1 = args["consider_m1"]

    # Random effect - aggregated GP 
    length = numpyro.sample("kernel_length", dist.InverseGamma(3,3))
    var = numpyro.sample("kernel_var",dist.HalfNormal(0.05))
    k = gp_kernel(x,x,var, length, noise)
    f = numpyro.sample(
        "f", 
        dist.MultivariateNormal(loc = jnp.zeros(x.shape[0]), covariance_matrix = k)
        )

    #aggregate f into gp_aggr according to indexing of (point in polygon)
    if consider_m1:
        print(colored(f"Aggregation - M1 : {M1.shape} , f : {f.shape}", "red"))
        gp_aggr_m1 = numpyro.deterministic("gp_aggr", M_g(M1, f))
        print(colored(f"gp_aggr.shape : {gp_aggr_m1.shape}", "red"))
        return gp_aggr_m1
    else:
        gp_aggr_m1 = numpyro.deterministic("gp_aggr_m1", M_g(M1, f))
        gp_aggr_m2 = numpyro.deterministic("gp_aggr_m2", M_g(M2, f))
        gp_aggr = numpyro.deterministic("gp_aggr", jnp.concatenate([gp_aggr_m1,gp_aggr_m2]))
        return gp_aggr
    
# -------------------------- Variational Autoencoder ------------------------- #
def vae_encoder(hidden_dim, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init = stax.randn()),
        stax.Elu,
        stax.FanOut(2),
        stax.parallel(
            # mean 
            stax.Dense(z_dim, W_init = stax.randn()),
            #std 
            stax.serial(stax.Dense(z_dim, W_init = stax.randn()), stax.Exp)
        )
    )

def vae_decoder(hidden_dim, out_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init = stax.randn()),
        stax.Elu,
        stax.Dense(out_dim, W_init = stax.randn())
    )

# def vae_model(batch, hidden_dim, z_dim): #lets assume batch here are the encodings
#     batch = jnp.reshape(batch, (batch.shape[0], -1)) #(*,2)
#     batch_dim, out_dim = jnp.shape(batch) #( * , 2)
#     decode = numpyro.module(
#         "decoder",
#         nn = vae_decoder(hidden_dim= hidden_dim, out_dim = out_dim),#(*,2)
#         input_shape = (batch_dim, z_dim) #(*,40)
#     )
#     #? Why do we sample here again ...
#     z = numpyro.sample("z", dist.Normal(jnp.zeros((z_dim,)), jnp.ones((z_dim,))))
#     gen_loc = decode(z)
#     numpyro.sample("obs", dist.Normal(gen_loc, args["vae_var"]), obs = batch)

def vae_model(batch, hidden_dim, z_dim):
    batch = jnp.reshape(batch, (batch.shape[0], -1)) # still gonna be (5,116)
    batch_dim, out_dim = jnp.shape(batch) # 5 , 116 

    # vae-decoder in numpyro module
    decode = numpyro.module(
        name = "decoder", 
        nn = vae_decoder(hidden_dim = hidden_dim, out_sim = out_dim),
        input_shape = (batch_dim, z_dim)    
    )

    # Sample a univariate normal
    z = numpyro.sample("z", dist.Normal(jnp.zeros((z_dim,)), jnp.ones((z_dim,)))) #(z_dim,) <- i.e (40,)
    # Forward pass from decoder
    gen_loc = decode(z) #(num_regions,) <- (116,)
    obs = numpyro.sample("obs", dist.Normal(gen_loc, args["vae_var"]), obs = batch) #(num_samples, num_regions) <- (5,116)
    return obs

# def vae_guide(batch, hidden_dim, z_dim):
#     batch = jnp.reshape(x, (x.shape[0], -1)) #(*,2)
#     batch_dim, input_dim = jnp.shape(batch) # (* , 2)
#     encode = numpyro.module(
#         name = "encoder", 
#         nn = vae_encoder(hidden_dim, z_dim), #(50,40)
#         input_shape = (batch_dim, input_dim) #(1920,2)
#     )
#     z_loc, z_std = encode(batch) #(*,40),(*,40)
#     z = numpyro.sample("z", dist.Normal(z_loc, z_std)) #(*,40)
#     return z

def vae_guide(batch, hidden_dim, z_dim):
    batch = jnp.reshape(batch, (batch.shape[0], -1)) #(num_samples, num_regions) <- (5,116)
    batch_dim, inpu_dim = jnp.shape(batch)# num_samples , num_regions <- 5 , 116 

    # vae-encoder in numpyro module
    encode = numpyro.module(
        name = "encoder", 
        nn = vae_encoder(hidden_dim=hidden_dim,z_dim = z_dim),
        input_shape = (batch_dim, input_dim)    
    ) 

    # Samapling mu, sigma - Pretty much the forward pass
    z_loc, z_std = encode(batch) #mu : (num_samples, z_dim), sigma2 : (num_samples, z_dim) <- (5,40),(5,40)

    z = numpyro.sample("z", dist.Normal(z_loc, z_std)) #(num_sample, z_dim) <- (5,40)
    return z


# def vae_encoder(hidden_dim, z_dim):
#     return stax.serial(
#         stax.Dense(hidden_dim, W_init=stax.randn()),
#         stax.Elu,
#         stax.FanOut(2),
#         stax.parallel(
#             stax.Dense(z_dim, W_init=stax.randn()), # mean
#             stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp), # std -- i.e. diagonal covariance
#         ),
#     )


# def vae_decoder( hidden_dim, out_dim):
#     return stax.serial(
#         stax.Dense(hidden_dim, W_init=stax.randn()),
#         stax.Elu,
#         stax.Dense(out_dim, W_init=stax.randn())
#     )

# def vae_model(batch, hidden_dim, z_dim):
#     batch = jnp.reshape(batch, (batch.shape[0], -1))
#     batch_dim, out_dim = jnp.shape(batch)
#     decode = numpyro.module("decoder", vae_decoder( hidden_dim, out_dim), (batch_dim, z_dim))
#     z = numpyro.sample("z", dist.Normal(jnp.zeros((z_dim,)), jnp.ones((z_dim,))))
#     gen_loc = decode(z)    
#     numpyro.sample("obs", dist.Normal(gen_loc, args["vae_var"]), obs=batch)
    

# def vae_guide(batch, hidden_dim,  z_dim):
#     batch = jnp.reshape(batch, (batch.shape[0], -1))
#     batch_dim, out_dim = jnp.shape(batch)
#     encode = numpyro.module("encoder", vae_encoder(hidden_dim, z_dim), (batch_dim, out_dim))
#     z_loc, z_std = encode(batch)
#     z = numpyro.sample("z", dist.Normal(z_loc, z_std))
#     return z

@jit 
def epoch_train(rng_key, svi_state, num_train):
    def body_fn(i, val):
        # Random keys
        rng_key_i = random.fold_in(rng_key, i)
        rng_key_i, rng_key_ls, rng_key_var, rng_key_noise = random.split(rng_key_i, 4)
        loss_sum, svi_state = val 
        # Gp Draw 
        batch = agg_gp_predictive(rng_key, args)["gp_aggr"] #(num_samples, num_regions) <- (5,116)
        # Forward pass, Takes "state" & "args" where args are your "model inputs" 
        svi_state, loss = svi.update(svi_state, batch)
        print(colored("check1", "green"))
        loss_sum += loss / args["batch_size"]
        return loss_sum, svi_state 
    
    return lax.fori_loop(lower =0, upper = num_train, body_fun = body_fn, init_val = (0.0, svi_state))

@jit
def eval_test(rng_key, svi_state, num_test):
    def body_fn(i, loss_sum):
        rng_key_i = random.fold_in(rng_key, i)
        rng_key_i, rng_key_ls, rng_key_var, rng_key_noise = random.split(rng_key_i, 4)
        batch = agg_gp_predictive(rng_key_i, args)["gp_aggr"] #(num_samples, num_regions) <- GP Draw
        loss = svi.evaluate(svi_state, batch)
        loss_sum += loss 
        return loss 
    loss = lax.fori_loop(0, num_test, body_fn, 0.0)
    loss = loss / num_test

    return loss

    
    

# -------------------- Function to plot a gaussian process ------------------- #
def plot_process(gp_draws):
    p = px.line()
    for i in range(len(gp_draws)):
        p.add_scatter(x = np.arange(gp_draws.shape[1]), y = gp_draws[i, :])

    p.update_traces(line_color = "black")
    p.update_layout(
        template = "plotly_white", 
        xaxis_title = "region", yaxis_title = "num cases",
        showlegend = False
    )
    p.show()


if __name__ == "__main__":
    # ----------------------------- Load data needed ----------------------------- #
    #todo Need to change these variables
    var_dir = "misc/interim_saved_data_var"
    x = np.load(os.path.join(var_dir, "x.npy"))
    pol_pt_new = np.load(os.path.join(var_dir, "pol_pt_new.npy"))
    pol_pt_old = np.load(os.path.join(var_dir, "pol_pt_old.npy"))
    pt_which_pol_new = np.load(os.path.join(var_dir, "pt_which_pol_new.npy"))
    pt_which_pol_old = np.load(os.path.join(var_dir, "pt_which_pol_old.npy"))
    s_new = gpd.read_file(os.path.join(var_dir, "s_new", "s_new.shp"))
    s_old = gpd.read_file(os.path.join(var_dir, "s_old", "s_old.shp"))
    s = gpd.read_file(os.path.join(var_dir, "old_new_geom_cases", "old_new_geom_cases.shp"))

    

    # --------------------------------- Arguments -------------------------------- #
    args = {'n_obs': jnp.array(s.n_obs),
        "x": x,
        "gp_kernel": exp_sq_kernel,
        "noise": 1e-4,
        "M1": pol_pt_old,
        "M2": pol_pt_new,

        # VAE training
        "rng_key": random.PRNGKey(5),
        "num_epochs": 20, 
        #"learning_rate": 1.0e-3, 
        "learning_rate": 0.0005, 
        "batch_size": 100, 
        "hidden_dim": 50, 
        "z_dim": 40, 
        "num_train": 100,
        "num_test":100,
        "vae_var": 1,

        # Consider One Region or Both
        "consider_m1" : False
    }

    # ------------------------ Prior Predictive Simulation ----------------------- #
    rng_key, rng_key_ = random.split(random.PRNGKey(4))
    agg_gp_predictive = Predictive(gp_aggr,num_samples = 5)
    consider_m1 = False 
    
    # Returns (n_Samples, num_regions_in_m1)
    agg_gp_draws = agg_gp_predictive(rng_key_, args)["gp_aggr"] #(num_sample, num_regions)
    
   
    # Plotting 
    #plot_process(agg_gp_draws)


    
    #-------------------------- Initiate Training Loop -------------------------- #
    adam = numpyro.optim.Adam(step_size = args["learning_rate"])
    svi = SVI(
        vae_model, 
        vae_guide, 
        adam, 
        RenyiELBO(), 
        hidden_dim = args["hidden_dim"], 
        z_dim = args["z_dim"]
    )

    rng_key, rng_key_samp, rng_key_init = random.split(args["rng_key"],3)
    init_batch = agg_gp_predictive(rng_key_, args)["gp_aggr"] #(num_samples, num_regions) i.e (5, 116)
    svi_state = svi.init(rng_key_init, init_batch)
    
    test_loss_list = []
    for i in range(args["num_epochs"]):
        rng_key, rng_key_train, rng_key_test, rng_key_infer = random.split(rng_key, 4)
        num_train = 1000
        # Where forward/backward pass gets called for train
        _ , svi_state = epoch_train(rng_key_train, svi_state, num_train)

        num_test = 1000 
        # Where forward/backward pass gets called for test
        test_loss = eval_test(rng_key_test, svi_state, num_test)
        test_loss_list += [test_loss]

        print("Epoch : {}, loss : {} ({:.2f} s.)".format(i, test_loss, test.time() - t_start))

        if math.isnan(test_loss):
            break 
    



    
     
    

