#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz
import xarray as xr
from scipy.interpolate import PchipInterpolator
from scipy.stats import linregress
import pytensor
# pytensor.config.gcc__cxxflags = '-L/opt/miniconda3/envs/bap3/lib -O3 -march=native'
# pytensor.config.cxx = '/usr/bin/clang++'
# pytensor.config.blas__ldflags = '-framework Accelerate'

def create_summary_table(summaries, model_labels=None):
    import pandas as pd

    if model_labels is None:
        model_labels = [f'Model_{i+1}' for i in range(len(summaries))]

    dfs = [df.T.rename(columns=lambda x: f'{x}_{label}') 
           for df, label in zip(summaries, model_labels)]

    final_df = pd.concat(dfs, axis=1)

    # Optional: Reorder the index
    index_order = ['mean', 'sd', 'hdi_3%', 'hdi_97%']
    final_df = final_df.reindex(index_order)

    # Optional: Round numerical values
    final_df = final_df.round(3)

    return final_df
#%%