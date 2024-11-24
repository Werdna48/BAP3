#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz
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

#%% Exercise 1
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

        idata_h = pm.sample(random_seed=4591)

    with pm.Model() as model_h_nh:
        #prior
        θ = pm.Beta('θ', alpha=1, beta=1, shape=len(N_samples))
        #likelihood
        y = pm.Bernoulli('y', p=θ[group_idx], observed=data)

        idata_nh = pm.sample(random_seed=4591)

    az.plot_forest([idata_h, idata_nh], model_names=['hierarchical', 'non-hierarchical'], var_names=['θ'], combined=True)

# %% Exercise 3
# Create a hierarchcical version of tips (pools over days of the week) and compare the results to the original

# tips = pd.read_csv("/Users/uqamcka3/PHD/bap_Martin/BAP3/code/data/tips.csv")
tips = pd.read_csv("D:\BAP\BAP3\code\data\\tips.csv")
categories = np.array(["Thur", "Fri", "Sat", "Sun"])

# Extract tip data and encode days into numerical indices
tip = tips["tip"].values
idx = pd.Categorical(tips["day"], categories=categories).codes

# Define coordinates for mapping categorical variables to the model
coords = {"days": categories, "days_flat":categories[idx]}

# NON-HIERARCHICAL MODEL
if __name__ == '__main__':
    # This model assumes no relationship between the days of the week.
    with pm.Model(coords=coords) as comparing_groups_nh:
        # Prior on group means for each day: Independent HalfNormal priors for each day
        mu = pm.HalfNormal("mu", sigma=5, dims="days")
        
        # Prior on group standard deviations for each day
        sigma = pm.HalfNormal("sigma", sigma=1, dims="days")
        
        # Likelihood: Observed tips are modeled as Gamma distributed
        # 'mu[idx]' and 'sigma[idx]' select the appropriate day-specific parameters
        y = pm.Gamma("y", mu=mu[idx], sigma=sigma[idx], observed=tip, dims="days_flat")
        
        # Perform sampling
        idata_cg_nh = pm.sample(2000, random_seed=4591)

    # HIERARCHICAL MODEL 00
    with pm.Model(coords=coords) as comparing_groups_h_00:
        # Hyperprior on group means (shared across days)
        mu_g = pm.Gamma("mu_g", mu=5, sigma=2)
        
        # Day-specific priors for group means, informed by the hyperprior
        mu = pm.HalfNormal("mu", sigma=mu_g, dims="days")
        
        # Day-specific priors for standard deviations
        sigma = pm.HalfNormal("sigma", sigma=1, dims="days")
        
        # Likelihood: Observed tips are Gamma distributed, similar to the non-hierarchical model
        y = pm.Gamma("y", mu=mu[idx], sigma=sigma[idx], observed=tip, dims="days_flat")
        
        # Perform sampling
        idata_cg_h_00 = pm.sample(random_seed=4591)

    # HIERARCHICAL MODEL 01
    with pm.Model(coords=coords) as comparing_groups_h_01:
        # Hyperpriors for group means and group standard deviations
        mu_g = pm.Gamma("mu_g", mu=5, sigma=2)       # Shared hyperprior for group means
        sigma_g = pm.Gamma("sigma_g", mu=2, sigma=1.5)  # Shared hyperprior for variability
        
        # Day-specific group means with Gamma prior, influenced by hyperpriors
        mu = pm.Gamma("mu", mu=mu_g, sigma=sigma_g, dims="days")
        
        # Day-specific standard deviations
        sigma = pm.HalfNormal("sigma", sigma=1, dims="days")
        
        # Likelihood: Observed tips are modeled as Gamma distributed
        y = pm.Gamma("y", mu=mu[idx], sigma=sigma[idx], observed=tip, dims="days_flat")
        
        # Perform sampling
        idata_cg_h_01 = pm.sample(random_seed=4591)

    # NON-CENTERED HIERARCHICAL MODEL 02
    # This is an alternative parameterization of the hierarchical model
    # Improves sampling efficiency by reparameterizing the group means
    with pm.Model(coords=coords) as comparing_groups_h_02:
        # Shared hyperpriors for group means and variability
        mu_g = pm.Gamma("mu_g", mu=5, sigma=2)
        sigma_g = pm.HalfNormal("sigma_g", sigma=2)
        
        # Non-centered parameterization for group means
        mu_g_offset = pm.Normal("mu_g_offset", sigma=2, dims="days")
        mu = pm.Deterministic("mu", mu_g + mu_g_offset * sigma_g, dims="days")
        
        # Day-specific standard deviations
        sigma = pm.HalfNormal("sigma", sigma=1, dims="days")
        
        # Likelihood: Observed tips as Gamma distributed
        y = pm.Gamma("y", mu=mu[idx], sigma=sigma[idx], observed=tip, dims="days_flat")
        
        # Perform sampling with a higher target_accept to handle potential divergences
        idata_cg_h_02 = pm.sample(random_seed=4591, target_accept=0.99)

    # Comparison: Visualize posterior distributions
    axes = az.plot_forest(
        [idata_cg_nh, idata_cg_h_00, idata_cg_h_01],
        model_names=['non_hierarchical', 'hierarchical_00', 'hierarchical_01'],
        var_names=['mu'], combined=True, r_hat=False, ess=False, figsize=(12, 3),
        colors='cycle'
    )
    y_lims = axes[0].get_ylim()
    axes[0].vlines(idata_cg_h_00.posterior['mu_g'].mean(), *y_lims, color="k", ls=":")

    # Compare "Thur - Fri" differences for all models
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.flatten()
    model_data = [
        ("Model NH", idata_cg_nh),
        ("Model H 00", idata_cg_h_00),
        ("Model H 01", idata_cg_h_01),
        ("Model H 02", idata_cg_h_02),
    ]

    for i, (model_name, idata) in enumerate(model_data):
        Ex3_posterior = az.extract(idata)
        means_diff = Ex3_posterior["mu"].sel(days="Thur") - Ex3_posterior["mu"].sel(days="Fri")
        az.plot_posterior(means_diff.values, ref_val=0, ax=axes[i])
        axes[i].set_title(f"Thur - Fri {model_name}")
    plt.show()

    # Summarize results for all models
    summaries = [az.summary(idata, kind="stats") for idata in [idata_cg_nh, idata_cg_h_00, idata_cg_h_01, idata_cg_h_02]]
    model_sums = create_summary_table(summaries)

# %% Exercise 4
# For each subpanel in fig 637 add a reference line representing the empirical mean value at each level
# Compare the empirical values to the posterior, what do you observe?
# FW mean seem overestimated but player seems good
football = pd.read_csv("D:\BAP\BAP3\code\data\\football_players.csv", dtype={'position':'category'})
football["gps"] = football["goals"] / football["shots"]

pos_idx = football.position.cat.codes.values
pos_codes = football.position.cat.categories
n_pos = pos_codes.size
n_players = football.index.size
print(pos_codes)
coords = {'pos': pos_codes}
print(coords)

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


    param_Est_football = az.summary(idata_football, kind="stats").round(2)

    _, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    ax[0].axvline(football["gps"].mean(), color="0.25", ls="--")
    az.plot_posterior(idata_football, var_names='mu', ax=ax[0])
    ax[0].set_title(r"Global mean")
    ax[1].axvline(football["gps"][football['position'] == 'FW'].mean(), color="0.25", ls="--")
    az.plot_posterior(idata_football.posterior.sel(pos="FW"), var_names='mu_p', ax=ax[1])
    ax[1].set_title(r"Forward position mean")
    ax[2].axvline(football["gps"][football['name'] == 'Lionel Messi'].values, color="0.25", ls="--")
    az.plot_posterior(idata_football.posterior.sel(theta_i_dim_0=1457), var_names='theta_i', ax=ax[2])
    ax[2].set_title(r"Messi mean")
    plt.show()
 
# %% Exercise 5 

# Build a hierarchical model including group effects for polar, non-polar, charged, and special
cs_data = pd.read_csv('D:\BAP\BAP3\code\data\chemical_shifts_theo_exp.csv')
# cs_data = pd.read_csv('/Users/uqamcka3/PHD/bap_Martin/BAP3/code/data/chemical_shifts_theo_exp.csv')
diff = cs_data.theo - cs_data.exp
cat_encode = pd.Categorical(cs_data['cat']) # Changed to cat to group by category
idx = cat_encode.codes
coords = {"cat": cat_encode.categories}

if __name__ == '__main__':
    with pm.Model(coords=coords) as cs_cat:
        #hyperpriors on mu
        mu_hyp = pm.Normal('mu_hyp', mu=0, sigma=10)
        sd_hyp = pm.HalfNormal('sd_hyp', 10)
        # The above says that the hyperperior of mu's is normal with some mean and sd

        #priors
        mu = pm.Normal('mu', mu=mu_hyp, sigma=sd_hyp, dims='cat') #Not the use of the hierachical priors for mean and sd at individual level
        sd = pm.HalfNormal('sd', sigma=10, dims='cat')

        #likelihood
        y = pm.Normal('y', mu=mu[idx], sigma=sd[idx], observed=diff)

        #Sampling
        idata_cs_cat = pm.sample(random_seed=4591)


    param_Esth = az.summary(idata_cs_cat, kind="stats").round(2)
    az.plot_trace(idata_cs_cat)
    plt.show()

    # Compare the models 
    axes = az.plot_forest([idata_cs_cat],
                        var_names='mu', combined=True, r_hat=False, ess=False, figsize=(10, 7),
                        colors='cycle')
    y_lims = axes[0].get_ylim()
    axes[0].vlines(idata_cs_cat.posterior['mu_hyp'].mean(), *y_lims, color="k", ls=":")
    plt.show()
# %%
