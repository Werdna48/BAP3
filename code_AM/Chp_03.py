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

#%% Important style code
az.style.use("arviz-grayscale")
from cycler import cycler
default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
plt.rc('axes', prop_cycle=default_cycler)
plt.rc('figure', dpi=300)
np.random.seed(123)

#%% Load theoretical NMR data
# cs_data = pd.read_csv('D:\BAP\BAP3\code\data\chemical_shifts_theo_exp.csv')
cs_data = pd.read_csv('/Users/uqamcka3/PHD/bap_Martin/BAP3/code/data/chemical_shifts_theo_exp.csv')
diff = cs_data.theo - cs_data.exp
cat_encode = pd.Categorical(cs_data['aa'])
idx = cat_encode.codes
coords = {"aa": cat_encode.categories}

# Run a model where instead of any hierarchy we fit each group independently 
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
    az.plot_trace(idata_cs_nh)
    plt.show()
# %%
