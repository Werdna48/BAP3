#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz
import pytensor
from scipy.special import erf
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
# Change the parameters of the prior Beta distribution in our_first_model
# to match those of the previous chapter. Compare the results to the previous chapter
np.random.seed(123)
trials = 4
theta_real = 0.35  # True (but unknown) probability of success
data = pz.Binomial(n=1, p=theta_real).rvs(trials)  # Simulating observed data
print("Simulated data:", data)

beta_params = [(1,1), (20, 20), (1, 4)] #Alpha first and beta second
idatas = []
if __name__ == '__main__':
    for alpha, beta in beta_params: 
        with pm.Model() as our_first_model:
            theta = pm.Beta('theta', alpha=alpha, beta=beta)  # Prior for θ
            y = pm.Bernoulli('y', p=theta, observed=data)  # Likelihood for observed data
            idata = pm.sample(1000, random_seed=4591, cores=4, chains=4)  # Posterior sampling
            idatas.append(idata)
    
    #Plot posterior for each model
    _, axes = plt.subplots(1,3, figsize=(12, 4))
    for idata, ax in zip(idatas, axes):
        az.plot_posterior(idata, ax=ax)
    plt.show()
    #Compare uncertainty around estimates
    az.plot_forest(idatas, combined=True, figsize=(10, 2))
    plt.show()

#%% Exercise 2
# Change the prior for theta in our first model to uniform(0,1)
# Are the posteriors different? Only the [-1,2] posterior is different but still centered appropriately
# Is the sampling slower, or fasters, or the same? Same for all
# What happens if you change the interval? divergences on [-1, 2] probabbly because the likelihood cant take negative parameters
# What errors did you get
np.random.seed(123)
trials = 4
theta_real = 0.35  # True (but unknown) probability of success
data = pz.Binomial(n=1, p=theta_real).rvs(trials)  # Simulating observed data
print("Simulated data:", data)
uniform_params = [(0,1), (-1, 2)]
idatasEx2 = []

if __name__ == '__main__':
    print('Fitting Beta')
    with pm.Model() as our_first_model:
        theta = pm.Beta('theta', alpha=1, beta=1)  # Prior for θ
        y = pm.Bernoulli('y', p=theta, observed=data)  # Likelihood for observed data
        idata = pm.sample(1000, random_seed=4591, cores=4, chains=4)  # Posterior sampling
        idatasEx2.append(idata)

    for lower, upper in uniform_params:
        print(f'Fitting Uniform {[lower, upper]}')
        with pm.Model() as our_first_model:
            theta = pm.Uniform('theta', lower=lower, upper=upper)  # Prior for θ
            y = pm.Bernoulli('y', p=theta, observed=data)  # Likelihood for observed data
            idata = pm.sample(1000, random_seed=4591, cores=4, chains=4)  # Posterior sampling
            idatasEx2.append(idata)

    #Plot posterior for each model
    _, axes = plt.subplots(1,3, figsize=(12, 4))
    for idata, ax in zip(idatasEx2, axes):
        az.plot_posterior(idata, ax=ax)
    plt.show()
    #Compare uncertainty around estimates
    az.plot_forest(idatasEx2, combined=True, figsize=(10, 2))
    plt.show()


# %% Exercise 5
# Modify model_g. Prior for the mean to a Gaussian centered at the empirical mean,
# Try three sets of parameters for the sd.
# How robust/sensitive are the infereneces to these changes? Very robust uncertainy increases with higher sd but they are all centered and have similar HDIs
# What do you think of using a gaussian which is unboundeded when the data are bound to [0, 100]: Using an unbounded distirbution should be fine as long as the data themselves are bounded

# Load Gaussian data (e.g., chemical shifts)
data = np.loadtxt("D:\BAP\BAP3\code\data\chemical_shifts.csv") #Windows
# data = np.loadtxt("/Users/uqamcka3/PHD/bap_Martin/BAP3/code/data/chemical_shifts.csv") #MAC
empirical_mean = np.mean(data).round(2)
sd_vals = [0.5, 1, 10]
idatasEx3 = []
#%%
if __name__ == '__main__':
    # Model specification and sampling for Gaussian inference
    for sigma in sd_vals:
        with pm.Model() as model_g:
            mu = pm.Normal(f'mu_p{sigma}', mu=empirical_mean, sigma=sigma)  # Prior for μ
            sigma = pm.HalfNormal('sigma', 5)  # Prior for σ

            Y = pm.Normal('Y', mu=mu, sigma=sigma, observed=data)  # Likelihood
            idata_g = pm.sample(random_seed=4591)  # Posterior sampling

            idatasEx3.append(idata_g)

    #Plot posterior for each model
    _, axes = plt.subplots(1,3, figsize=(12, 4))
    for idata, ax in zip(idatasEx3, axes):
        az.plot_posterior(idata, ax=ax)
    plt.show()
    #Compare uncertainty around estimates
    az.plot_forest(idatasEx3, combined=True, figsize=(10, 2))
    plt.show()

    #Summaries
    summaries = [az.summary(idata, kind="stats") for idata in idatasEx3]
    model_labels = [f'Prior_{i+1}' for i in range(len(summaries))]

    df = create_summary_table(summaries, model_labels)

# %% Exercise 6
# Using the data from chem shifts, compute the mean and sd with and without outliers
# Copmare results to the bayesian estimation using Gaussian and students t, what do you observe?
# The bayesian results are similar to the empirical means of the data!!
copy_data = data.copy()
empirical_mean = np.mean(copy_data).round(2)
empirical_sd = np.std(copy_data).round(2)

outlier_mask = ~(np.abs(data-empirical_mean) < empirical_sd*2)
copy_data[outlier_mask]
empirical_mean_no_outliers = np.mean(copy_data[~outlier_mask])
empirical_std_no_outliers = np.std(copy_data[~outlier_mask])

idatasEx6 = []

#Now run the models
if __name__ == '__main__':
    # Model specification and sampling for Gaussian inference
    with pm.Model() as model_g:
        mu = pm.Uniform('mu', lower=40, upper=70)  # Prior for μ
        sigma = pm.HalfNormal('sigma', 5)  # Prior for σ

        Y = pm.Normal('Y', mu=mu, sigma=sigma, observed=data)  # Likelihood
        idata_g = pm.sample(random_seed=4591, cores=4, chains=4)  # Posterior sampling
        idatasEx6.append(idata_g)

    with pm.Model() as model_tr:
        # Priors
        mu = pm.Uniform('mu', lower=40, upper=75)  # Prior for μ
        sigma = pm.HalfNormal('sigma', sigma=10)  # Prior for σ
        nu = pm.Exponential('nu', 1 / 30)  # Prior for ν (degrees of freedom)

        # Likelihood
        y = pm.StudentT('y', nu=nu, mu=mu, sigma=sigma, observed=data)  # Robust likelihood

        # Sampling from posterior
        idata_tr = pm.sample(random_seed=4591, cores=4, chains=4)
        idatasEx6.append(idata_tr)
    
    #Plot posterior for each model
    _, axes = plt.subplots(1,2, figsize=(12, 4))
    for idata, ax in zip(idatasEx6, axes):
        az.plot_posterior(idata, ax=ax)
    plt.show()
    #Compare uncertainty around estimates
    az.plot_forest(idatasEx6, combined=True, figsize=(10, 2))
    plt.show()

    #Summaries
    summaries = [az.summary(idata, kind="stats") for idata in idatasEx6]
    model_sums = create_summary_table(summaries)
    print(f'Outlier {(empirical_mean, empirical_sd)}', '\n', f'NoOutlier: {(empirical_mean_no_outliers, empirical_std_no_outliers)}')
# %% Exercise 7
# Repeat the previous but add more outliers to chemical shifts and compute new posteriors
#  what do you find? With a small amount of outliers proportional to your data the gaussian t is good at keeping the same results
copy_data = data.copy()
# Create an array with the outliers repeated 4 times
additional_outliers = np.repeat(data[outlier_mask], 4)

# Concatenate the original data array and the additional outliers
data_more_outliers = np.concatenate([data, additional_outliers])
empirical_mean = np.mean(data_more_outliers).round(2)
empirical_sd = np.std(data_more_outliers).round(2)


idatasEx7 = []

#Now run the models
if __name__ == '__main__':
    # Model specification and sampling for Gaussian inference
    with pm.Model() as model_g:
        mu = pm.Uniform('mu', lower=40, upper=70)  # Prior for μ
        sigma = pm.HalfNormal('sigma', 5)  # Prior for σ

        Y = pm.Normal('Y', mu=mu, sigma=sigma, observed=data_more_outliers)  # Likelihood
        idata_g = pm.sample(random_seed=4591, cores=4, chains=4)  # Posterior sampling
        idatasEx7.append(idata_g)

    with pm.Model() as model_tr:
        # Priors
        mu = pm.Uniform('mu', lower=40, upper=75)  # Prior for μ
        sigma = pm.HalfNormal('sigma', sigma=10)  # Prior for σ
        nu = pm.Exponential('nu', 1 / 30)  # Prior for ν (degrees of freedom)

        # Likelihood
        y = pm.StudentT('y', nu=nu, mu=mu, sigma=sigma, observed=data_more_outliers)  # Robust likelihood

        # Sampling from posterior
        idata_tr = pm.sample(random_seed=4591, cores=4, chains=4)
        idatasEx7.append(idata_tr)
    
    #Plot posterior for each model
    _, axes = plt.subplots(1,2, figsize=(12, 4))
    for idata, ax in zip(idatasEx7, axes):
        az.plot_posterior(idata, ax=ax)
    plt.show()
    #Compare uncertainty around estimates
    az.plot_forest(idatasEx7, combined=True, figsize=(10, 2))
    plt.show()

    #Summaries
    summaries = [az.summary(idata, kind="stats") for idata in idatasEx7]
    model_sums = create_summary_table(summaries)
    print(f'Outlier {(empirical_mean, empirical_sd)}')


# %% Exercise 8
# Explore the inferencedata object for idata_cg
# How many groups does it have? 4
# Inspect the posterior using sel method
# Compute the distributions of mean differences between thrusday and sunday. 
#   What are the coordinates and dimensions of the reuslting array
tips = pd.read_csv('D:\\BAP\\BAP3\\code\data\\tips.csv')

#Create variable representing tips in dollars
categories = np.array(["Thur", "Fri", 'Sat', 'Sun'])
tip = tips['tip'].values 
# Create index variable that encodes days with numbers
# [0, 1, 2, 3] instead of [’Thur’,’Fri’, ’Sat’, ’Sun’].
idx = pd.Categorical(tips['day'], categories=categories).codes
#In special cases we can specify coordinates to keep labels through the fitting proces
coords = {'days': categories, 'days_flat': categories[idx]}
# THis dictonary is actually a dictionary that holds one list called days which contains the labels

if __name__ == '__main__':
    with pm.Model(coords=coords) as comparing_groups: # NOTE THE COORDS MAPPPING HERE
        # We cange from shape to dims, not sure why
        #priors
        mu = pm.HalfNormal('mu', sigma=5, dims='days') #Priors are VECTORS not Scalars
        sigma = pm.HalfNormal('sigma', sigma=1, dims='days')

        #Likelihood
        #use our idx variable to pass our 4 categories into the 4 vector positions
        y = pm.Gamma('y', mu=mu[idx], sigma=sigma[idx], observed=tip, dims='days_flat') #for some reason days flat here?

        #sample the model
        idata_cg = pm.sample(chains=4, cores=4, random_seed=4591)
        #add posterior predictive
        idata_cg.extend(pm.sample_posterior_predictive(idata_cg, random_seed=4591))

        # Compute the distributions of mean differences between thrusday and sunday
        posterior = az.extract(idata_cg.posterior)
        TH_SdiffDist = posterior['mu'].sel(days='Thur') - posterior['mu'].sel(days='Sun')
        az.plot_kde(TH_SdiffDist.values)
        plt.show()
        # Exercise 9
        # Compute the probability of superiority directly from the posterior.
        # Get our model's predictions for possible tips on Thursday and Sunday Flatten over all dimensions
        posterior_predictive_thursday =  idata_cg.posterior_predictive["y"].sel(days_flat="Thur").stack(samples=("chain", "draw", "days_flat"))
        posterior_predictive_sunday = idata_cg.posterior_predictive["y"].sel(days_flat="Sun").stack(samples=("chain", "draw", "days_flat"))

        az.plot_kde(posterior_predictive_thursday.values, label="Thursday")
        az.plot_kde(posterior_predictive_sunday.values, plot_kwargs={"color":"C1"}, label="Sunday")
        plt.show() 
        #the logic here is now we know how much overlap between the distributions is just though random draws from empirical distirbutions
        thursday_tip_draws = np.random.choice(posterior_predictive_thursday, replace=True, size=1000)
        sunday_tip_draws = np.random.choice(posterior_predictive_sunday, replace=True, size=1000)

        (thursday_tip_draws > sunday_tip_draws).mean()
        print((thursday_tip_draws > sunday_tip_draws).mean())

        #Calculate cohens d for fun
        dist = pz.Normal(0, 1)

        means_diff = posterior['mu'].sel(days='Thur') -  posterior['mu'].sel(days='Sun')
        
        d_cohen = (means_diff /
                np.sqrt((posterior["sigma"].sel(days='Thur')**2 + 
                            posterior["sigma"].sel(days='Sun')**2) / 2)
                ).mean().item()
        
        ps = dist.cdf(d_cohen/(2**0.5))
        print(f"Cohens d: {d_cohen}")
        print(ps, (thursday_tip_draws > sunday_tip_draws).mean())
# %%
