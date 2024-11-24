#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz
import pytensor

# Only for Mac
# pytensor.config.gcc__cxxflags = '-L/opt/miniconda3/envs/bap3/lib -O3 -march=native'
# pytensor.config.cxx = '/usr/bin/clang++'
# pytensor.config.blas__ldflags = '-framework Accelerate'

#%% Important style code
az.style.use("arviz-grayscale")
from cycler import cycler
default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
plt.rc('axes', prop_cycle=default_cycler)
plt.rc('figure', dpi=100, figsize=(10, 6))
np.random.seed(123)

#%% Load theoretical NMR data
cs_data = pd.read_csv('D:\BAP\BAP3\code\data\chemical_shifts_theo_exp.csv')
# cs_data = pd.read_csv('/Users/uqamcka3/PHD/bap_Martin/BAP3/code/data/chemical_shifts_theo_exp.csv')
diff = cs_data.theo - cs_data.exp
cat_encode = pd.Categorical(cs_data['aa'])
idx = cat_encode.codes
coords = {"aa": cat_encode.categories}

## Now for our modelling we have multiple decisions
# Hierarchical - Each amino acids are a fanmily of compounds so we assume they are all the same and estimate a single gaussian for all the differences
# Non-Hierarchical - Each amino acid is different (there are 20 unique ones) so a better choice might be to fit 20 separate gaussians

#%%
# Run a model where instead of any hierarchy we fit each group independently (Unpooled) 
if __name__ == '__main__':
    with pm.Model(coords=coords) as cs_nh:
        #priors
        mu = pm.Normal('mu', mu=0, sigma=10, dims='aa')
        sd = pm.HalfNormal('sd', sigma=10, dims='aa')

        #likelihood
        y = pm.Normal('y', mu=mu[idx], sigma=sd[idx], observed=diff)

        #Sampling
        idata_cs_nh = pm.sample(random_seed=4591)


    param_Est = az.summary(idata_cs_nh, kind="stats").round(2)
    print(param_Est)
    az.plot_trace(idata_cs_nh)
    plt.show()

# Run a model where we add a hierarchy (Hyper parameter)
# This parameter is the distribution that represents the distribution of means of amino acids generally

    with pm.Model(coords=coords) as cs_h:
        #hyperpriors on mu
        mu_hyp = pm.Normal('mu_hyp', mu=0, sigma=10)
        sd_hyp = pm.HalfNormal('sd_hyp', 10)
        # The above says that the hyperperior of mu's is normal with some mean and sd

        #priors
        mu = pm.Normal('mu', mu=mu_hyp, sigma=sd_hyp, dims='aa') #Not the use of the hierachical priors for mean and sd at individual level
        sd = pm.HalfNormal('sd', sigma=10, dims='aa')

        #likelihood
        y = pm.Normal('y', mu=mu[idx], sigma=sd[idx], observed=diff)

        #Sampling
        idata_cs_h = pm.sample(random_seed=4591)


    param_Esth = az.summary(idata_cs_h, kind="stats").round(2)
    print(param_Esth)
    az.plot_trace(idata_cs_h)
    plt.show()

    # Compare the models 
    axes = az.plot_forest([idata_cs_nh, idata_cs_h], model_names=['non_hierarchical', 'hierarchical'],
                        var_names='mu', combined=True, r_hat=False, ess=False, figsize=(10, 7),
                        colors='cycle')
    y_lims = axes[0].get_ylim()
    axes[0].vlines(idata_cs_h.posterior['mu_hyp'].mean(), *y_lims, color="k", ls=":")
    plt.show()
# %% Water Quality Example
# Analyse quality of water in a city, 
# samples taken by dividing city into neighborhoods
# We have two things we can do:
# - Study each neighbourhood as a seperate entity
# - Pool all the data together and estimate teh water quality of the city as a single big group

# By using a pooled (Hierarchical) Model we can do both (estimate the neigbourhoods and the group level)

# Simulate the data
# 3 Neighbourhoods

N_samples = [30, 30, 30]  # Number of samples collected in each group/neighborhood
# Good (1) is below WHO lead recommendations, Bad (0) is above 
G_samples = [18, 18, 18]  # Number of good quality water samples in each neighborhood

group_idx = np.repeat(np.arange(len(N_samples)), N_samples)  
# Generate a list of group indices where each sample belongs (e.g., group 0, 1, or 2)
# `np.arange(len(N_samples))` creates indices [0, 1, 2] for the three neighborhoods.
# `np.repeat` replicates each index based on the number of samples in that group.

data = []  # Initialize an empty list to store the data (binary values: 1 for good water, 0 for bad water)
for i in range(0, len(N_samples)):  # Loop over the number of groups (neighborhoods)
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i] - G_samples[i]]))
    # For each group, create a list of 1s (good water samples) and 0s (bad water samples)
    # `G_samples[i]` specifies the number of good samples
    # `N_samples[i] - G_samples[i]` specifies the number of bad samples
    # `np.repeat([1, 0], [good_count, bad_count])` creates a list with the specified number of 1s and 0s
    # `data.extend` appends this list to the main `data` list

# %% Modelling
# We will be using this model
# Likelihood : y_i ~ Bernoulli(theta_i)
# Prior : theta_i ~ beta(mu, nu)
# Hyperprior on nu : nu ~ HalfNormal(sigma_nu)
# Hyperprior on mu : mu ~ Beta(alpha_mu, beta_mu)

if __name__ == '__main__':
    with pm.Model() as model_h:
        #hyper priors
        mu = pm.Beta('mu', 1, 1) #City level (group) mean
        nu = pm.HalfNormal('nu', 10) #City level (group) concentration

        #priors
        theta = pm.Beta('theta', mu=mu, nu=nu, shape=len(N_samples)) # A theta per neighbour hood

        #likelihood
        y = pm.Bernoulli('y', p=theta[group_idx], observed=data)

        idata_h = pm.sample()

    param_Est_water = az.summary(idata_h, kind="stats").round(2)
    print(param_Est_water)
    az.plot_trace(idata_h)
    plt.show()

    posterior = az.extract(idata_h, num_samples=100)
    for sample in posterior[["mu", "nu"]].to_array().values.T:
        pz.Beta(mu=sample[0], kappa=sample[1]).plot_pdf(legend=None, color="C0", alpha=0.1, support=(0.01, 0.99), moments="m")

    pz.Beta(mu=posterior["mu"].mean().item(), kappa=posterior["nu"].mean().item()).plot_pdf(legend=None, color="C0", moments="m")
    plt.xlabel('$theta_{prior}$')
# %% Soccer Player Positions
# Premier League data set
# Interested in goals per shot (success rate)

# Model :
#Likelihood : gs ~ Binomial(theta_i)

#Prior on theta_i : theta _i ~ Beta(theta_pos)
# Beta(theta_pos) = Beta(mu_p, nu_p)

# Hyperprior on mu_p : mu_p ~ beta(mu, v)
# Hyperprior on mu ~ beta(1.7, 5.8)
# Hyperprior on nu ~ gamma(mu=125, sigma=50)
football = pd.read_csv("D:\BAP\BAP3\code\data\\football_players.csv", dtype={'position':'category'})
print(football.head)

pos_idx = football.position.cat.codes.values
pos_codes = football.position.cat.categories
n_pos = pos_codes.size
n_players = football.index.size
print(pos_codes)
coords = {'pos': pos_codes}
print(coords)
# %% Modelling
if __name__ == "__main__":
    with pm.Model(coords=coords) as model_football:
        # Hyperpriors
        nu = pm.Gamma('nu', mu=125, sigma=50)
        mu = pm.Beta('mu', 1.7, 5.8)

        # Postition Parameters on theta_p for theta_i (success rate %)
        # vector of 4 mu's for each pos DF, FW, GK, MF
        mu_p = pm.Beta('mu_p', mu=mu, nu=nu, dims='pos')
        nu_p = pm.Gamma('nu_p', mu=125, sigma=50, dims='pos')

        #Player parameters
        #pos idx is how we match the vector of 4 to the same value in our data chat gpt explain this more and give me an example how how this linking is working with 2 minimal reproducable examples
        theta_i = pm.Beta('theta_i', mu=mu_p[pos_idx], nu=nu_p[pos_idx])

        # Likelihood which we seem not to care about? Why is this?
        _ = pm.Binomial('gs', n=football.shots.values, p=theta_i, observed=football.goals.values)
        
        idata_football = pm.sample()


    print(idata_football.posterior)
    param_Est_football = az.summary(idata_football, kind="stats").round(2)
    print(param_Est_football)

    _, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    az.plot_posterior(idata_football, var_names='mu', ax=ax[0])
    ax[0].set_title(r"Global mean")
    az.plot_posterior(idata_football.posterior.sel(pos="FW"), var_names='mu_p', ax=ax[1])
    ax[1].set_title(r"Forward position mean")
    az.plot_posterior(idata_football.posterior.sel(theta_i_dim_0=1457), var_names='theta_i', ax=ax[2])
    ax[2].set_title(r"Messi mean")
    plt.show()

    az.plot_forest(idata_football, var_names=['mu_p'], combined=True, figsize=(12, 3))
    plt.show()
# %%
