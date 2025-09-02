# Generative Modelling to estimate Influenza prevalence in High resolution regions using coarse resolution data

- The following will use a Gaussian Process to model spatial correlation and estimate influenza prevelce using Beta-Binomial Distribution - Note the approach is too slow as the grid size increases (To speed up use VAE approach mentioned below.)
    - `python src/py/aggGPv2.py`
    - Arguments
        - `--data_path` : path to data folder and saved vectors. 
        - `--n_warmup` : Number of samples to warmup MCMC
        - `--n_samples` : Number of MCMC posterior samples

- Train a Variational Autoencoder (VAE) to approximate computationally expensive Gaussian Proceesses (GP) to speed up MCMC sampling.
    - `python src/py/aggVAEv2.py`
    - Arguments 
        - `--data_path` : Path to data folder and saved vectors
        - `--hidden_dim` : VAE hidden dimension
        - `--z_dim` : latent vector dimension 
        - `--n_samples` : number of GP samples 
        - `--n_batches` : number of batches 
        - `--epochs` : number of epochs 
        - `--gen_gp_on_fly` : generate stochastic samples of GPs every epoch - Model will see different realizations of GP epochs. Dont use the flag if you want to see the same samples every batch. 

- Downstream influenza prevelence estimation - Use trained decorder to generate GP's to speed up MCMC sampling when estimating prevalence. 
    - `python src/py/aggVAEPreVv2.py` 
    - Arguments
        - `--data_path` : Path to data folder and saved vectors
        - `--vae_pretrain_rt` : Folder to save VAE weights
        - `--n_warmup` : number of warmup samples for MCMC
        - `--n_samples` : number of MCMC posterior samples 
        - `--ignore_mu` : consider only random effects and covariates and avoid fixed effects.