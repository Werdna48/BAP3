#%% [markdown]
# This script demonstrates Bayesian inference for Bernoulli and Gaussian models using PyMC.
# It ties each modeling step and visualization to concepts covered in Chapter 2.

#%%
# Essential imports for Bayesian analysis with PyMC and visualization tools.
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz
import pytensor

# PyTensor configuration for performance
pytensor.config.gcc__cxxflags = '-L/opt/miniconda3/envs/bap3/lib -O3 -march=native'
pytensor.config.cxx = '/usr/bin/clang++'
pytensor.config.blas__ldflags = '-framework Accelerate'

# Setting consistent sampling configurations: cores=4, chains=4, and a fixed random seed
# ensures reproducibility, as recommended in Chapter 2.

#%%
# Configure ArviZ and Matplotlib styles for better visuals
az.style.use("arviz-grayscale")
from cycler import cycler
default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
plt.rc('axes', prop_cycle=default_cycler)
plt.rc('figure', dpi=300)

#%% [markdown]
# **Simulating Data**:
# The Bernoulli process simulates trials with a binary outcome based on `theta_real`.
# This provides the observed data for inference.

#%%
np.random.seed(123)
trials = 4
theta_real = 0.35  # True (but unknown) probability of success
data = pz.Binomial(n=1, p=theta_real).rvs(trials)  # Simulating observed data
print("Simulated data:", data)

#%% [markdown]
# **Model Specification for Bernoulli Process**:
# - **Prior**: The Beta distribution (`Beta(1, 1)`) represents our initial belief about θ.
# - **Likelihood**: The Bernoulli distribution links θ to the observed data (`data`).
# - **Inference**: Sampling uses Markov Chain Monte Carlo (MCMC) to estimate the posterior.

#%%
if __name__ == '__main__':
    with pm.Model() as our_first_model:
        θ = pm.Beta('θ', alpha=1., beta=1.)  # Prior for θ
        y = pm.Bernoulli('y', p=θ, observed=data)  # Likelihood for observed data
        idata = pm.sample(1000, random_seed=4591, cores=4, chains=4, mp_ctx="fork")  # Posterior sampling

#%% [markdown]
# **Inference Data Object**:
# - `idata` stores all results from the model:
#   - **posterior**: Samples of θ.
#   - **sample_stats**: Diagnostics like effective sample size and convergence statistics.
#   - **observed_data**: Original data used in the likelihood.
# - This structure allows seamless exploration and diagnostics using ArviZ.

#%%
# Trace plot to visualize parameter convergence across chains
az.plot_trace(idata)
plt.title("Trace Plot for Bernoulli Model")
plt.show()

#%%
# Summary of posterior samples for Bernoulli model
print("Posterior summary for Bernoulli model:")
print(az.summary(idata, kind="stats").round(2))

#%% [markdown]
# **Gaussian Inference**:
# We now perform inference on Gaussian-distributed data (e.g., chemical shifts from NMR).
# The Gaussian model specifies:
# - **Prior for μ**: Uniform distribution over plausible range of means.
# - **Prior for σ**: HalfNormal distribution for standard deviation.
# - **Likelihood**: Gaussian, linking μ and σ to observed data.

#%%
# Load Gaussian data (e.g., chemical shifts)
data = np.loadtxt("/Users/uqamcka3/PHD/bap_Martin/BAP3/code/data/chemical_shifts.csv")
_, ax = plt.subplots(figsize=(12, 3))
ax.boxplot(data, vert=False)
plt.title("Boxplot of Observed Gaussian Data")
plt.show()

#%%
# Model specification and sampling for Gaussian inference
if __name__ == '__main__':
    with pm.Model() as model_g:
        mu = pm.Uniform('mu', lower=40, upper=70)  # Prior for μ
        sigma = pm.HalfNormal('sigma', 5)  # Prior for σ

        Y = pm.Normal('Y', mu=mu, sigma=sigma, observed=data)  # Likelihood
        idata_g = pm.sample(random_seed=4591, cores=4, chains=4, mp_ctx="fork")  # Posterior sampling

#%% [markdown]
# **Gaussian Inference Data Object**:
# - The Gaussian model's `idata_g` contains:
#   - **posterior**: Samples of μ and σ.
#   - **posterior predictive**: Predictions based on the posterior.
#   - **sample_stats**: Diagnostics.
#   - **observed_data**: Original chemical shifts data.

#%%
# Trace plot for Gaussian model
az.plot_trace(idata_g)
plt.show()

#%%
# Summary of posterior for Gaussian model
print("Posterior summary for Gaussian model:")
print(az.summary(idata_g, kind="stats").round(2))

#%% [markdown]
# **Posterior Predictive Checks for Gaussian Model**:
# Posterior predictive sampling evaluates how well the model captures observed data.

#%%
# Generate posterior predictive samples and extend `idata_g`
pm.sample_posterior_predictive(idata_g, model=model_g, extend_inferencedata=True)

#%% [markdown]
# - `idata_g` now includes:
#   - **posterior predictive**: Simulated data based on posterior parameter samples.
#   - Can be accessed via `idata_g.posterior_predictive`.

#%%
# Posterior Predictive Check visualization
az.plot_ppc(idata_g, num_pp_samples=100)
plt.title("Posterior Predictive Check for Gaussian Model")
plt.show()

#%% [markdown]
# **Robust Gaussian Inference**:
# We now perform robust inference on Gaussian-distributed data (e.g., chemical shifts from NMR).
# The Gaussian model specifies:
# - **Prior for μ**: Uniform distribution over plausible range of means.
# - **Prior for σ**: HalfNormal distribution for standard deviation.
# - **Pripr for v**: Exponential for normality/degrees of freedom - helps to correct for outliers
# - **Likelihood**: Students T distribution, linking μ, σ, and v to observed data.
# %%
if __name__ == '__main__':
    with pm.Model() as model_t:
        #priors
        mu = pm.Uniform('mu', 40, 75)
        sigma = pm.HalfNormal('sigma', sigma=10)
        v = pm.Exponential('v', 1/30)
        #likelihood
        y = pm.StudentT('y', nu=v, mu=mu, sigma=sigma, observed=data)
        #sampling
        idata_t = pm.sample(random_seed=4591, mp_ctx="fork")

#%% [markdown]
# **Robust Gaussian Inference**:
# We now perform robust inference on Gaussian-distributed data (e.g., chemical shifts from NMR).
# The robust Gaussian model handles outliers better by using a Student's T distribution.
# - **Prior for μ**: Uniform distribution over a plausible range of means.
# - **Prior for σ**: HalfNormal distribution for the standard deviation.
# - **Prior for ν (degrees of freedom)**: Exponential distribution to control the heaviness of tails.
# - **Likelihood**: Student's T distribution, linking μ, σ, and ν to observed data.

#%%
# Model specification and sampling for robust Gaussian inference
if __name__ == '__main__':
    with pm.Model() as model_tr:
        # Priors
        mu = pm.Uniform('mu', lower=40, upper=75)  # Prior for μ
        sigma = pm.HalfNormal('sigma', sigma=10)  # Prior for σ
        nu = pm.Exponential('nu', 1 / 30)  # Prior for ν (degrees of freedom)

        # Likelihood
        y = pm.StudentT('y', nu=nu, mu=mu, sigma=sigma, observed=data)  # Robust likelihood

        # Sampling from posterior
        idata_tr = pm.sample(random_seed=4591, cores=4, chains=4, mp_ctx="fork")

#%% [markdown]
# **Robust Gaussian Inference Data Object**:
# - `idata_t` stores all results from the model:
#   - **posterior**: Samples of μ, σ, and ν.
#   - **posterior predictive**: Predictions based on posterior samples.
#   - **sample_stats**: Diagnostics like effective sample size and convergence statistics.
#   - **observed_data**: Original data used in the likelihood.
# - This structure supports seamless exploration and diagnostics with ArviZ.

#%%
# Trace plot for robust Gaussian model
az.plot_trace(idata_tr)
plt.title("Trace Plot for Robust Gaussian Model")
plt.show()

#%%
# Summary of posterior samples for the robust Gaussian model
print("Posterior summary for robust Gaussian model:")
print(az.summary(idata_tr, kind="stats").round(2))

#%% [markdown]
# **Posterior Predictive Checks for Robust Gaussian Model**:
# Posterior predictive sampling evaluates how well the robust model captures the observed data.

#%%
# Generate posterior predictive samples and extend `idata_t`
pm.sample_posterior_predictive(idata_tr, model=model_tr, extend_inferencedata=True)

#%% [markdown]
# - `idata_t` now includes:
#   - **posterior predictive**: Simulated data based on posterior parameter samples.
#   - Accessible via `idata_t.posterior_predictive`.

#%%
# Posterior Predictive Check visualization for robust Gaussian model
az.plot_ppc(idata_t, num_pp_samples=100)
plt.title("Posterior Predictive Check for Robust Gaussian Model")
plt.show()

#%% [markdown]
# - `idata_t` now includes:
#   - **posterior predictive**: Simulated data based on posterior parameter samples.
#   - Accessible via `idata_t.posterior_predictive`.

#%% [markdown]
# **InferenceData Object**:
# We can access the posterior as follows:
posterior = idata_g.posterior #('This is an xarray dataset')
# To get the first draw from chain 0 and chain 2 we can:
posterior.sel(draw=0, chain=[0,2])

#if we want the first 100 draws from all chains we have:
posterior.sel(draw=slice(0,100))

#We can get the means for all parameters over all draws and chains as:
posterior.mean()

#
