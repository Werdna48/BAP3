#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz
import pytensor
import os
# pytensor.config.gcc__cxxflags = '-L/opt/miniconda3/envs/bap3/lib -march=native' 
# pytensor.config.cxx = '/usr/bin/clang++'

#%%
az.style.use("arviz-grayscale")
from cycler import cycler
default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
plt.rc('axes', prop_cycle=default_cycler)
plt.rc('figure', dpi=300)

#%%
np.random.seed(123)
trials = 4
theta_real = 0.35 # unknown value in a real experiment
data = pz.Binomial(n=1, p=theta_real).rvs(trials)
data

#%%

if __name__ == '__main__':
    with pm.Model() as our_first_model:
        θ = pm.Beta('θ', alpha=1., beta=1.)
        y = pm.Bernoulli('y', p=θ, observed=data)
        idata = pm.sample(1000, random_seed=4591)
# %%
