#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz
import xarray as xr

np.random.seed(123)
az.style.use("arviz-doc")

howell = pd.read_csv("D:\BAP\BAP3\code\data\howell.csv", delimiter=";")
howell_age_18 =  howell[howell["age"] > 18]
howell_age_18 = howell_age_18.reset_index()
# %% Exercise 1 Linear regresssion of weight vs height
if __name__ == '__main__':
    with pm.Model() as how_o18:
        # Define mutable data for weight to make predictions
        x_weight = pm.MutableData('x_weight', howell_age_18['weight'])
        
        #priors
        alpha = pm.Normal('alpha', mu = 0, sigma = 10)
        beta = pm.Normal('beta', mu = 0, sigma = 10)
        sigma = pm.HalfNormal('sigma', 10)

        #Linear model
        mu = pm.Deterministic('mu', alpha + beta * x_weight)

        #Likelihood
        y_pred = pm.Normal('y_pred', mu=mu, sigma=sigma, observed=howell_age_18['height'], shape=x_weight.shape[0])

        #Sampling
        idata_o18 = pm.sample(random_seed=4591)

        pm.sample_posterior_predictive(idata_o18, model=how_o18, extend_inferencedata=True)
    
    az.plot_trace(idata_o18, var_names=['~mu'])
    
    az.plot_posterior(idata_o18, var_names=['~mu'])

    posterior = az.extract(idata_o18, num_samples=50)

    estimates = az.summary(idata_o18, kind='stats')

    # Lets start the plotting
    x_plot = xr.DataArray(np.linspace(howell_age_18['weight'].min(), howell_age_18['weight'].max(), 50), dims='pos_idx')

    #Lets get the mean line using the mean of alpha and beta times all our x's
    mean_line = posterior['alpha'].mean() + posterior['beta'].mean() * x_plot

    #Sample some lines from the posterior
    sample_line = posterior['alpha'] + posterior['beta'] * x_plot
    #Get the HDI of these lines
    hdi_line = az.hdi(idata_o18.posterior['mu'])

    #Let's set up the actual plots now :D
    fig, axes = plt.subplots(1, 2)

    #left plot is posterior lines
    axes[0].scatter(howell_age_18['weight'], howell_age_18['height']) # Scatter plot of data
    lines_ = axes[0].plot(x_plot, sample_line.T, alpha=0.2, label='Sample Lines') # Posterior lines
    plt.setp(lines_[1:], label="_")  # Hide extra labels
    axes[0].plot(x_plot, mean_line, label="Mean Line")  # Mean prediction line
    axes[0].set_xlabel("Weight")
    axes[0].set_ylabel("Height")
    axes[0].legend()

    # Right plot: 94% HDI for predictions to show uncertainty as a shaded region
    axes[1].scatter(howell_age_18['weight'], howell_age_18['height'])  # Scatter plot of data
    idx = np.argsort(howell_age_18['weight'])  # Sort temperatures for plotting HDI
    axes[1].fill_between(
        howell_age_18.weight[idx],
        hdi_line["mu"][:, 0].values[idx],
        hdi_line["mu"][:, 1].values[idx],
        color="C1",
        label="HDI",
        alpha=0.5,
    )  # HDI shaded area
    axes[1].plot(x_plot, mean_line, label="mean line")  # Mean prediction line
    axes[1].set_xlabel("Weight")
    axes[1].legend()

    # Display the plots
    plt.show()
    # Exercise 2 - Given weights 45.73, 65.8, 54.2, 32.59 predict their heights

    weights = (45.73, 65.8, 54.2, 32.59)
    with how_o18:
        how_o18.set_data('x_weight', weights)
        ppc = az.extract(pm.sample_posterior_predictive(idata_o18), group="posterior_predictive")

    for idx, w in enumerate(weights):
        ppc_one = ppc["y_pred"].sel(y_pred_dim_2=idx).values
        #KDE shows all uncertainty
        ax = az.plot_kde(ppc_one, plot_kwargs={"color": f"C{idx}"}, label=f"{w} kg")
    ax.legend()
    ax.set_title(f"Posterior predictive distribution new data points")
# %% Exercise 3/4 Original dataset log weight
if __name__ == '__main__':
    with pm.Model() as how:
        # Define mutable data for weight to make predictions
        x_weight = pm.MutableData('x_weight', howell['weight'])
        
        #priors
        alpha = pm.Normal('alpha', mu = 0, sigma = 10)
        beta = pm.Normal('beta', mu = 0, sigma = 10)
        gamma = pm.HalfNormal('gamma', 10)
        delta = pm.HalfNormal('delta', 10)

        #Linear model
        mu = pm.Deterministic('mu', alpha + beta * x_weight**0.5)
        sigma = pm.Deterministic('sigma', gamma + delta * x_weight)

        #Likelihood
        y_pred = pm.Normal('y_pred', mu=mu, sigma=sigma, observed=howell['height'], shape=x_weight.shape[0])

        #Sampling
        idata_how = pm.sample(random_seed=4591)

        pm.sample_posterior_predictive(idata_how, model=how, extend_inferencedata=True)
    
    az.plot_trace(idata_how, var_names=['~mu', '~sigma'])
    
    az.plot_posterior(idata_how, var_names=['~mu', '~sigma'])

    posterior = az.extract(idata_how, num_samples=50)

    estimates = az.summary(idata_how, kind='stats')

    # Lets start the plotting
    x_plot = xr.DataArray(np.linspace(howell['weight'].min(), howell['weight'].max(), 50), dims='pos_idx')

    #Lets get the mean line using the mean of alpha and beta times all our x's
    mean_line = posterior['alpha'].mean() + posterior['beta'].mean() * x_plot**0.5

    #Sample some lines from the posterior
    sample_line = posterior['alpha'] + posterior['beta'] * x_plot**0.5
    #Get the HDI of these lines
    hdi_line = az.hdi(idata_how.posterior_predictive['y_pred'])

    #Let's set up the actual plots now :D
    fig, axes = plt.subplots(1, 2)

    #left plot is posterior lines
    axes[0].scatter(howell['weight'], howell['height']) # Scatter plot of data
    lines_ = axes[0].plot(x_plot, sample_line.T, alpha=0.2, label='Sample Lines') # Posterior lines
    plt.setp(lines_[1:], label="_")  # Hide extra labels
    axes[0].plot(x_plot, mean_line, label="Mean Line")  # Mean prediction line
    axes[0].set_xlabel("Weight")
    axes[0].set_ylabel("Height")
    axes[0].legend()

    # Right plot: 94% HDI for predictions to show uncertainty as a shaded region
    axes[1].scatter(howell['weight'], howell['height'])  # Scatter plot of data
    idx = np.argsort(howell['weight'])  # Sort temperatures for plotting HDI
    axes[1].fill_between(
        howell.weight[idx],
        hdi_line["y_pred"][:, 0].values[idx],
        hdi_line["y_pred"][:, 1].values[idx],
        color="C1",
        label="HDI",
        alpha=0.5,
    )  # HDI shaded area
    axes[1].plot(x_plot, mean_line, label="mean line")  # Mean prediction line
    axes[1].set_xlabel("Weight")
    axes[1].legend()

    # Display the plots
    plt.show()
# %% Exercise 5
ans = pd.read_csv('D:\BAP\BAP3\code\data\\anscombe.csv')

x_4 = ans[ans.group == 'IV']['x'].values
y_4 = ans[ans.group == 'IV']['y'].values
if __name__ == '__main__':
    with pm.Model() as model_t2_exp:
        α = pm.Normal('α', mu=0, sigma=100)
        β = pm.Normal('β', mu=0, sigma=1)
        ϵ = pm.HalfCauchy('ϵ', 5)
        ν = pm.Exponential('ν', 1/30)
        #ν = pm.Gamma('ν', mu=20, sigma=15)
        #ν = pm.Gamma('ν', 2, 0.1)

        y_pred = pm.StudentT('y_pred', mu=α + β * x_4, sigma=ϵ, nu=ν, observed=y_4)
        idata_t2_exp = pm.sample_prior_predictive()
        idata_t2_exp.extend(pm.sample())

    with pm.Model() as model_t2_gamma:
        α = pm.Normal('α', mu=0, sigma=100)
        β = pm.Normal('β', mu=0, sigma=1)
        ϵ = pm.HalfCauchy('ϵ', 5)
        # ν = pm.Exponential('ν', 1/30)
        # ν = pm.Gamma('ν', mu=20, sigma=15)
        ν = pm.Gamma('ν', 2, 0.1)

        y_pred = pm.StudentT('y_pred', mu=α + β * x_4, sigma=ϵ, nu=ν, observed=y_4)
        idata_t2_gamma = pm.sample_prior_predictive()
        idata_t2_gamma.extend(pm.sample())

    az.plot_forest([idata_t2_exp, idata_t2_gamma], model_names=["Exponential", "Gamma"], var_names=["~ν"], figsize=(8, 3.5), combined=True)

    az.plot_forest([idata_t2_exp, idata_t2_gamma], model_names=["Exponential", "Gamma"], var_names=["ν"], figsize=(8, 3.5), combined=True)

    az.plot_dist_comparison(idata_t2_exp, var_names=["ν"], figsize=(10, 4))

    az.plot_dist_comparison(idata_t2_gamma, var_names=["ν"], figsize=(10, 4))
# %% Exercise 6/7 re run model with some diff vars (7 is with student t instead of normal)

# Load the Iris dataset
iris = pd.read_csv("D:/BAP/BAP3/code/data/iris.csv")

# Filter dataset to only two species for binary classification
df = iris.query("species == ('setosa', 'versicolor')")

idatas = []
nus = [1,10,30]

vars = ['sepal_length','petal_length', 'petal_width']
if __name__ == '__main__':
    for nu in nus:
        # Encode species as binary values: setosa=0, versicolor=1
        y_0 = pd.Categorical(df["species"]).codes

        # Feature for classification
        x_n = 'petal_length'

        # Extract feature values and center them for numerical stability
        x_0 = df[x_n].values
        x_c = x_0 - x_0.mean()  # Centering helps sampling efficiency in Bayesian models
        with pm.Model() as model_irs:
            # Define priors for the intercept (alpha) and slope (beta) of the logistic model
            # Prior reflects beliefs about parameters before observing data
            alpha = pm.StudentT('alpha', mu=0, nu=nu, sigma=1)  # Prior for the intercept
            beta = pm.StudentT('beta', mu=0, nu=nu, sigma=5)    # Prior for the slope

            # Define the linear model for log-odds
            # mu = log-odds of being versicolor
            mu = alpha + x_c * beta

            # Transform log-odds to probability (theta) using the logistic (sigmoid) function
            # theta = 1 / (1 + exp(-mu)), which maps real numbers to the range [0, 1]
            theta = pm.Deterministic('theta', pm.math.sigmoid(mu))

            # Compute the decision boundary: the x_c value where the probability = 0.5
            # This is where alpha + beta * x_c = 0, hence x_c = -alpha / beta
            bd = pm.Deterministic('bd', -alpha / beta)

            # Define likelihood: observed binary outcomes modeled as Bernoulli trials
            # with success probability theta
            yl = pm.Bernoulli('yl', p=theta, observed=y_0)

            # Perform posterior sampling to estimate parameter distributions
            idata_lrs = pm.sample(random_seed=4591)

            idatas.append(idata_lrs)

        # Visualize posterior distributions for model parameters
        az.plot_trace(idata_lrs, var_names=["~bd", "~theta"])

        az.plot_posterior(idata_lrs, var_names=["~theta"])

        # Extract posterior samples
        posterior = idata_lrs.posterior

        # Compute the mean predicted probability (theta) over the posterior samples
        theta_mean = posterior["theta"].mean(("chain", "draw"))

        # Sort the centered x values for smooth plotting of the logistic curve
        idx = np.argsort(x_c)

        # Create a figure for the logistic regression plot
        _, ax = plt.subplots(figsize=(12, 6))

        # Plot the logistic curve
        # x-axis: centered sepal length; y-axis: predicted probability of being versicolor
        ax.plot(x_c[idx], theta_mean[idx], color="C0", lw=2,
                label="Mean predicted probability (logistic curve)")

        # Plot the decision boundary as a vertical line
        # The decision boundary is the mean of the posterior for bd
        bd_mean = posterior["bd"].mean(("chain", "draw"))
        ax.vlines(bd_mean, 0, 1, color="C2", zorder=0, label="Decision boundary")

        # Add the HDI (highest density interval) for the decision boundary
        # Represents uncertainty in the boundary due to posterior distributions
        bd_hdi = az.hdi(posterior["bd"])
        ax.fill_betweenx([0, 1], bd_hdi["bd"][0], bd_hdi["bd"][1], color="C2", alpha=0.6, lw=0,
                        label="Decision boundary (HDI)")

        # Scatter plot of the actual data points
        # Jitter added to y-axis for better visualization of binary outcomes
        ax.scatter(x_c, np.random.normal(y_0, 0.02), marker=".", 
                color=[f"C{x}" for x in y_0], label="Data points")

        # Plot the HDI for the predicted probabilities
        # Shows the uncertainty in the logistic curve predictions
        az.plot_hdi(x_c, posterior["theta"], color="C0", ax=ax,
                    fill_kwargs={"alpha": 0.2, "label": "Prediction uncertainty (HDI)"})

        # Add labels and format the x-axis to original var scale
        ax.set_xlabel(x_n)
        ax.set_ylabel("Predicted probability (θ)", rotation=0)
        ax.legend()

        # Adjust x-axis ticks to match the original scale
        locs, _ = plt.xticks()
        ax.set_xticks(locs)
        ax.set_xticklabels(np.round(locs + x_0.mean(), 1))

        plt.title(f"Logistic Regression: Petal Length vs Probability of Being Versicolor")
        plt.show()

    
    var_names = ['alpha', 'beta', 'bd']

    for feature, idata in zip(nus, idatas):
        print(f"Feature {feature} summary")
        print(az.summary(idata, var_names=var_names, kind='stats'))

    var_names = ['alpha', 'beta', 'bd']
    az.plot_forest(idatas, model_names=vars, var_names=var_names, combined=True)
    plt.show()
# %%
