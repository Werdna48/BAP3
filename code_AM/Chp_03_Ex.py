#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz
import pytensor
pytensor.config.gcc__cxxflags = '-L/opt/miniconda3/envs/bap3/lib -O3 -march=native'
pytensor.config.cxx = '/usr/bin/clang++'
pytensor.config.blas__ldflags = '-framework Accelerate'
#%%
# Model h generation 

N_samples = [30, 30, 30]
G_samples = [18, 3, 3]
group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))

if __name__ == '__main__':
    with pm.Model() as model_h:
        # hypyerpriors
        μ = pm.Beta('μ', 1., 1.)
        ν = pm.HalfNormal('ν', 10)
        # prior
        θ = pm.Beta('θ', mu=μ, nu=ν, shape=len(N_samples))
        # likelihood
        y = pm.Bernoulli('y', p=θ[group_idx], observed=data)

        idata_h = pm.sample(random_seed=4591, mp_ctx="fork")

#%%
if __name__ == '__main__':
    with pm.Model() as model_h_flat:
        #prior
        θ = pm.Beta('θ', alpha=1, beta=1, shape=len(N_samples))
        #likelihood
        y = pm.Bernoulli('y', p=θ[group_idx], observed=data)

        idata_h_flat = pm.sample(random_seed=4591, mp_ctx="fork")

# %% Compare the hierarchical and non-hierarchical models
az.plot_forest([idata_h, idata_h_flat], model_names=['hierarchical', 'non-hierarchical'], var_names=['θ'], combined=True)

# %% Tips dataset from chapter 2 non hierarchical and hierarchical versions

tips = pd.read_csv("/Users/uqamcka3/PHD/bap_Martin/BAP3/code/data/tips.csv")
categories = np.array(["Thur", "Fri", "Sat", "Sun"])

tip = tips["tip"].values
idx = pd.Categorical(tips["day"], categories=categories).codes

coords = {"days": categories, "days_flat":categories[idx]}
if __name__ == '__main__':
    with pm.Model(coords=coords) as tipgroups:
        μ = pm.HalfNormal("μ", sigma=5, dims="days")
        σ = pm.HalfNormal("σ", sigma=1, dims="days")

        y = pm.Gamma("y", mu=μ[idx], sigma=σ[idx], observed=tip, dims="days_flat")

        idata_tg = pm.sample(random_seed=4591, cores=4, chains=4, mp_ctx="fork")


if __name__ == '__main__':
    with pm.Model(coords=coords) as tipspooled:
        #hyperpriors
        μ_g = pm.Gamma('μ_g', mu=5, sigma = 2)
        σ_g = pm.Gamma('σ_g', mu=1, sigma=1)
        #priors
        μ = pm.HalfNormal("μ", sigma=μ_g, dims="days")
        σ = pm.HalfNormal("σ", sigma=σ_g, dims="days")
        #likelihood
        y = pm.Gamma("y", mu=μ[idx], sigma=σ[idx], observed=tip, dims="days_flat")

        idata_tpool = pm.sample(random_seed=4591, cores=4, chains=4, mp_ctx="fork")

# %% Compare the hierarchical and non-hierarchical models
az.plot_forest([idata_tpool, idata_tg], model_names=['hierarchical', 'non-hierarchical'], combined=True)


# %%
