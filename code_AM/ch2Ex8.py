#%%
# Essential imports for Bayesian analysis with PyMC and visualization tools.
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz
import pytensor
#%%

tips = pd.read_csv('D:\\BAP\\BAP3\\code\\data\\tips.csv')
categories = np.array(["Thur", "Fri", "Sat", "Sun"])

tip = tips["tip"].values
idx = pd.Categorical(tips["day"], categories=categories).codes

#%%
coords = {'days':categories, 'days_flat': categories[idx]}

if __name__ =='__main__':
    with pm.Model(coords=coords) as copmaring_groups:
        #prior
        mu = pm.HalfNormal("mu", sigma=5, dims="days")
        sigma = pm.HalfNormal("sigma", sigma=1, dims="days")
        #likelihood
        y = pm.Gamma("y", mu=mu[idx], sigma=sigma[idx], observed=tip, dims="days_flat")

        idata_cg = pm.sample(random_seed=4591)
        idata_cg.extend(pm.sample_posterior_predictive(idata_cg, random_seed=4591))

#%%
if __name__ =='__main__':
    #Examine the number of groups of idatacg
    idata_cg

    #%% looks a posterior of mu for a specific day using sel
    idata_cg.posterior['mu']

    idata_cg.posterior['mu'].sel(days='Fri')

    # Average estimated Mu for friday
    idata_cg.posterior['mu'].sel(days='Fri').mean()

    #%% Compute differences between thursday and sunday
    # take the differences of distributions
    idata_cg.posterior['mu'].sel(days='Thur') - idata_cg.posterior['mu'].sel(days='Sun')
# %%
if __name__ =='__main__':
    # Extract posterior samples for Thursday and Sunday
    mu_thur = idata_cg.posterior['mu'].sel(days='Thur').values.flatten()
    mu_sun = idata_cg.posterior['mu'].sel(days='Sun').values.flatten()

    # Compute the difference between Thursday and Sunday
    mu_diff = mu_thur - mu_sun

    # Plot the KDEs
    plt.figure(figsize=(10, 6))

    # KDE for Thursday
    az.plot_kde(mu_thur, label="Thursday", plot_kwargs={"color": "blue", "linestyle": "-", "linewidth": 2})

    # KDE for Sunday
    az.plot_kde(mu_sun, label="Sunday", plot_kwargs={"color": "orange", "linestyle": "--", "linewidth": 2})

    # KDE for the difference
    az.plot_kde(mu_diff, label="Thursday - Sunday", plot_kwargs={"color": "green", "linestyle": "-.", "linewidth": 2})

    # Customize the plot
    plt.title("Posterior Distributions of Thursday, Sunday, and Their Difference")
    plt.xlabel("Estimated Tip Amount (mu)")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.show()