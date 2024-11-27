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

# %% Simple linear regression

# Importing necessary libraries and reading the dataset
# The dataset contains temperature and bike rental data from a city
bikes = pd.read_csv("D:\\BAP\\BAP3\\code\\data\\bikes.csv")

# Plotting the relationship between temperature and number of rented bikes
# This provides a visual understanding of how temperature might influence bike rentals
bikes.plot(x="temperature", y="rented", figsize=(12, 3), kind="scatter")

# The dataset contains 17379 records with 17 variables, but for this analysis,
# only 359 records and two variables ('temperature' and 'rented') are used.

# Mathematical model setup:
# We aim to model the relationship:
# rented_bikes ~ temperature
# Using a simple linear regression model:
#   y = α + β * x 
# where:
#   α (alpha): Intercept (expected number of bikes rented when temperature is 0)
#   β (beta): Slope (rate of increase in bike rentals per degree increase in temperature)
#   ε: Residual error, assumed to follow a Normal distribution with standard deviation σ

# In Bayesian terms:
# Priors:
#   α ~ Normal(0, 100)      # Broad prior, assuming α can be any value within a wide range
#   β ~ Normal(0, 10)       # Assumes slope is smaller but centered around 0
#   σ ~ HalfCauchy(10)      # Half-Cauchy prior for standard deviation (non-negative)
# Likelihood:
#   y_observed ~ Normal(μ, σ)
#   μ = α + β * temperature

# NOTE: The key concept is that the *mean* number of rented bikes is modeled as a 
# linear function of temperature.

if __name__ == '__main__':
    with pm.Model() as model_lbx:
        # Priors: Define prior distributions for α, β, and σ
        alpha = pm.Normal('alpha', mu=0, sigma=100)  # Prior for intercept
        beta = pm.Normal('beta', mu=0, sigma=10)     # Prior for slope
        sigma = pm.HalfCauchy('sigma', beta=10)      # Prior for standard deviation (positive values only)

        # Linear deterministic model for the mean (μ = α + β * temperature)
        mu = pm.Deterministic('mu', alpha + beta * bikes.temperature)

        # Likelihood: Observed data is normally distributed around the linear model μ
        y_pred = pm.Normal('y_pred', mu=mu, sigma=sigma, observed=bikes.rented)

        # Posterior sampling to estimate model parameters
        idata_lb = pm.sample(random_seed=4591)

    # Plotting posterior distributions for the parameters (excluding 'mu' to reduce clutter)
    # az.plot_posterior provides marginal distributions for posterior samples
    az.plot_posterior(idata_lb, var_names=['~mu'])

    # We interpret the posterior to understand the marginal distributions of:
    # - α: Expected rentals at 0°C
    # - β: Increase in rentals per degree of temperature
    # - σ: Uncertainty in bike rentals due to factors not explained by temperature

    # Visualizing uncertainty in predictions:
    # Extract 50 posterior samples to illustrate prediction uncertainty
    posterior = az.extract(idata_lb, num_samples=50)

    # Generate a range of temperatures for plotting the model
    x_plot = xr.DataArray(
        np.linspace(bikes.temperature.min(), bikes.temperature.max(), 50), dims='plot_id'
    )

    # Mean prediction line using posterior means of α and β
    mean_line = posterior['alpha'].mean() + posterior['beta'].mean() * x_plot

    # 50 sampled regression lines from posterior samples of α and β
    lines = posterior['alpha'] + posterior['beta'] * x_plot

    # Calculate 94% HDI (Highest Density Interval) for 'mu'
    # This provides credible intervals for predictions
    hdi_lines = az.hdi(idata_lb.posterior['mu'])

    # Plot the data and regression lines to visualize model fit and uncertainty
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Left plot: 50 posterior regression lines to show variability
    axes[0].plot(bikes.temperature, bikes.rented, "C2.", zorder=-3)  # Scatter plot of data
    lines_ = axes[0].plot(x_plot, lines.T, c="C1", alpha=0.2, label="lines")  # Posterior lines
    plt.setp(lines_[1:], label="_")  # Hide extra labels
    axes[0].plot(x_plot, mean_line, c="C0", label="mean line")  # Mean prediction line
    axes[0].set_xlabel("temperature")
    axes[0].set_ylabel("rented bikes")
    axes[0].legend()

    # Right plot: 94% HDI for predictions to show uncertainty as a shaded region
    axes[1].plot(bikes.temperature, bikes.rented, "C2.", zorder=-3)  # Scatter plot of data
    idx = np.argsort(bikes.temperature.values)  # Sort temperatures for plotting HDI
    axes[1].fill_between(
        bikes.temperature[idx],
        hdi_lines["mu"][:, 0][idx],
        hdi_lines["mu"][:, 1][idx],
        color="C1",
        label="HDI",
        alpha=0.5,
    )  # HDI shaded area
    axes[1].plot(x_plot, mean_line, c="C0", label="mean line")  # Mean prediction line
    axes[1].set_xlabel("temperature")
    axes[1].legend()

    # Display the plots
    plt.show()
    # %% Interpreting the posterior predictions
    pm.sample_posterior_predictive(idata_lb, model=model_lbx, extend_inferencedata=True)
    mean_line = idata_lb.posterior["mu"].mean(("chain", "draw"))
    temperatures = np.random.normal(bikes.temperature.values, 0.01)
    idx = np.argsort(temperatures)
    x = np.linspace(temperatures.min(), temperatures.max(), 15)
    y_pred_q = idata_lb.posterior_predictive["y_pred"].quantile(
        [0.03, 0.97, 0.25, 0.75], dim=["chain", "draw"]
    )
    y_hat_bounds = iter(
        [
            PchipInterpolator(temperatures[idx], y_pred_q[i][idx])(x)
            for i in range(4)
        ]
    )

    _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bikes.temperature, bikes.rented, "C2.", zorder=-3)
    ax.plot(bikes.temperature[idx], mean_line[idx], c="C0")

    for lb, ub in zip(y_hat_bounds, y_hat_bounds):
        ax.fill_between(x, lb, ub, color="C1", alpha=0.5)

    plt.show()
# %% Generalised linear model

# Importing the dataset
# This dataset contains temperature and bike rental data in a city
bikes = pd.read_csv("D:\\BAP\\BAP3\\code\\data\\bikes.csv")

# GLM setup:
# The generalized linear model allows us to use different distributions for the likelihood.
# In this case, we use a Negative Binomial distribution, suitable for count data.

# Mathematical model:
# Priors:
#   α ~ Normal(0, 1)        # Intercept prior
#   β ~ Normal(0, 10)       # Slope prior
#   σ ~ HalfNormal(10)      # Dispersion parameter prior (for overdispersion in count data)
# Deterministic transformation:
#   μ = exp(α + β * temperature)   # Transform the linear model output to ensure positivity
# Likelihood:
#   y_observed ~ NegativeBinomial(μ, α_dispersion)
# The Negative Binomial distribution is parameterized by:
# - Mean (μ): Expected number of rentals
# - Dispersion (α_dispersion): Controls variance (variance = μ + μ² / α_dispersion)

# NOTE: The exponential function ensures the mean μ remains positive, as required by the Negative Binomial distribution.
# Phi is negative binomial
# theta is dispersion (sigma)
# f is the exponential function
if __name__ == '__main__':
    # Linear model with Normal likelihood
    with pm.Model() as model_lb:
        alpha = pm.Normal('alpha', mu=0, sigma=100)  # Prior for intercept
        beta = pm.Normal('beta', mu=0, sigma=10)     # Prior for slope
        sigma = pm.HalfCauchy('sigma', beta=10)      # Prior for standard deviation (positive values only)

        mu = pm.Deterministic('mu', alpha + beta * bikes.temperature)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=sigma, observed=bikes.rented)

        idata_lb = pm.sample(random_seed=123)
        idata_lb.extend(pm.sample_posterior_predictive(idata_lb, random_seed=123))  # Add posterior predictive samples

    # Generalized linear model with Negative Binomial likelihood
    with pm.Model() as model_neg:
        alpha = pm.Normal("alpha", mu=0, sigma=1)                # Prior for intercept
        beta = pm.Normal("beta", mu=0, sigma=10)               # Prior for slope
        sigma = pm.HalfNormal("sigma", 10)                       # Prior for dispersion parameter

        mu = pm.Deterministic("mu", pm.math.exp(alpha + beta * bikes.temperature))  # Transform to positive μ
        y_pred = pm.NegativeBinomial("y_pred", mu=mu, alpha=sigma, observed=bikes.rented)

        idata_neg = pm.sample(random_seed=123)
        idata_neg.extend(pm.sample_posterior_predictive(idata_neg, random_seed=123))  # Add posterior predictive samples

    # Plotting posterior predictive for Negative Binomial
    mean_line = idata_neg.posterior["mu"].mean(("chain", "draw"))
    temperatures = np.random.normal(bikes.temperature.values, 0.01)  # Add jitter for smooth plotting
    idx = np.argsort(temperatures)
    x = np.linspace(temperatures.min(), temperatures.max(), 15)

    y_pred_q = idata_neg.posterior_predictive["y_pred"].quantile(
        [0.03, 0.97, 0.25, 0.75], dim=["chain", "draw"]
    )
    y_hat_bounds = iter(
        [
            PchipInterpolator(temperatures[idx], y_pred_q[i][idx])(x)
            for i in range(4)
        ]
    )

    # Create plot for posterior predictive check (Negative Binomial)
    _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bikes.temperature, bikes.rented, "C2.", zorder=-3)  # Observed data
    ax.plot(bikes.temperature[idx], mean_line[idx], c="C0")     # Posterior mean

    for lb, ub in zip(y_hat_bounds, y_hat_bounds):
        ax.fill_between(x, lb, ub, color="C1", alpha=0.5)       # Quantile bounds

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Number of Rented Bikes")
    plt.close()

    # Posterior predictive checks for both models
    _, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Normal likelihood model
    az.plot_ppc(
        idata_lb,
        num_pp_samples=200,
        alpha=0.1,
        colors=["C1", "C0", "C0"],
        ax=ax[0],
        mean=False,
    )
    ax[0].set_title("Normal")

    # Negative Binomial likelihood model
    az.plot_ppc(
        idata_neg,
        num_pp_samples=200,
        alpha=0.1,
        colors=["C1", "C0", "C0"],
        ax=ax[1],
        mean=False,
    )
    ax[1].set_title("NegativeBinomial")

# %% Robust Regression
# Importing Anscombe dataset
# This dataset is used to demonstrate how a Student's t-distribution improves robustness against outliers.
ans = pd.read_csv('D:\BAP\BAP3\code\data\\anscombe_3.csv')

if __name__ == '__main__':
    with pm.Model() as model_t:
        # Priors:
        # α ~ Normal(mean_y, 1): Prior for intercept, centered around mean of y.
        alpha = pm.Normal('alpha', mu=ans.y.mean(), sigma=1)
        
        # β ~ Normal(0, 1): Prior for slope, assuming a mild relationship between x and y.
        beta = pm.Normal('beta', mu=0, sigma=1)
        
        # σ ~ HalfNormal(5): Prior for standard deviation.
        sigma = pm.HalfNormal('sigma', 5)
        
        # ν_ ~ Exponential(1/29): Prior for degrees of freedom (influence of heavy tails).
        nu_ = pm.Exponential('nu_', 1/29)
        nu = pm.Deterministic('nu', nu_ + 1)  # Shifted to avoid values near 0.
        
        # Deterministic transformation:
        # μ = α + β * x: Linear relationship for the mean response.
        mu = pm.Deterministic('mu', alpha + beta * ans.x)
        
        # Likelihood:
        # y_observed ~ TruncatedStudentT(μ, σ, ν): Models outliers using heavy-tailed Student's t-distribution.
        _ = pm.ExGaussian('y_pred', mu=mu, sigma=sigma, nu=nu, observed=ans.y)
        
        # Sampling posterior distributions:
        idata_t = pm.sample(2000)

    # Plot trace of sampled parameters:
    az.plot_trace(idata_t, var_names=["~mu"])

    # Compare non-robust and robust regression fits:
    beta_c, alpha_c, *_ = linregress(ans.x, ans.y)

    _, ax = plt.subplots()
    ax.plot(ans.x, (alpha_c + beta_c * ans.x), "C0:", label="non-robust")  # Linear regression fit
    ax.plot(ans.x, ans.y, "C0o")
    alpha_m = idata_t.posterior["alpha"].mean(("chain", "draw"))
    beta_m = idata_t.posterior["beta"].mean(("chain", "draw"))

    # Generate fitted line from robust regression:
    x_plot = xr.DataArray(np.linspace(ans.x.min(), ans.x.max(), 50), dims="plot_id")
    ax.plot(x_plot, alpha_m + beta_m * x_plot, c="C0", label="robust")
    az.plot_hdi(ans.x, az.hdi(idata_t.posterior["mu"])["mu"].T, ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0)
    ax.legend(loc=2)

    # Posterior predictive checks:
    pm.sample_posterior_predictive(idata_t, model=model_t, random_seed=2, extend_inferencedata=True)
    ax = az.plot_ppc(idata_t, num_pp_samples=200, figsize=(12, 6), colors=["C1", "C0", "C1"])

#%% Logistic Regression
# Load the Iris dataset
iris = pd.read_csv("D:/BAP/BAP3/code/data/iris.csv")

# Filter dataset to only two species for binary classification
df = iris.query("species == ('setosa', 'versicolor')")

# Encode species as binary values: setosa=0, versicolor=1
y_0 = pd.Categorical(df["species"]).codes

# Feature for classification
x_n = "sepal_length"

# Extract feature values and center them for numerical stability
x_0 = df[x_n].values
x_c = x_0 - x_0.mean()  # Centering helps sampling efficiency in Bayesian models

if __name__ == '__main__':
    with pm.Model() as model_irs:
        # Define priors for the intercept (alpha) and slope (beta) of the logistic model
        # Prior reflects beliefs about parameters before observing data
        alpha = pm.Normal('alpha', mu=0, sigma=1)  # Prior for the intercept
        beta = pm.Normal('beta', mu=0, sigma=5)    # Prior for the slope

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

    # Add labels and format the x-axis to original sepal length scale
    ax.set_xlabel(x_n)
    ax.set_ylabel("Predicted probability (θ)", rotation=0)
    ax.legend()

    # Adjust x-axis ticks to match the original scale
    locs, _ = plt.xticks()
    ax.set_xticks(locs)
    ax.set_xticklabels(np.round(locs + x_0.mean(), 1))

    plt.title("Logistic Regression: Sepal Length vs Probability of Being Versicolor")
    plt.show()
# %% Variable Variance
# This model extends linear regression by allowing the standard deviation (SD) of the response variable
# to vary as a function of the predictor variable.

# Load the dataset containing infant growth data.
data = pd.read_csv('D:/BAP/BAP3/code/data/babies.csv')

# Visualize the data to observe trends between month (predictor) and length (response).
data.plot.scatter("month", "length")
plt.show()

# Model_vv has variable variance
# Model_nv has non-variable variance

if __name__ == '__main__':
    with pm.Model() as model_vv:
        # Define 'month' as shared mutable data for flexibility in posterior predictive sampling.
        x_shared = pm.MutableData('x_shared', data.month.values.astype(float))

        # Priors for the linear model of the mean (μ):
        # α (intercept): Represents the estimated average length at month=0.
        alpha = pm.Normal('alpha', sigma=10)

        # β (slope): Represents the rate of change in mean length as a function of sqrt(month).
        beta = pm.Normal('beta', sigma=10)

        # Priors for the linear model of the standard deviation (σ):
        # γ (intercept): Base variance when month=0.
        gamma = pm.HalfNormal('gamma', sigma=10)

        # δ (slope): The rate of change in variance with respect to month.
        delta = pm.HalfNormal('delta', sigma=10)

        # Linear model for mean (μ): μ = α + β * sqrt(month)
        # Using the square root of the predictor to account for potential non-linear relationships.
        mu = pm.Deterministic('mu', alpha + beta * x_shared**0.5)

        # Linear model for standard deviation (σ): σ = γ + δ * month
        # Allows σ to vary linearly with month, addressing heteroskedasticity.
        sigma = pm.Deterministic('sigma', gamma + delta * x_shared)

        # Likelihood function: Observed length is modeled as a normal distribution.
        # The mean and standard deviation are modeled as functions of the predictor variable.
        y_pred = pm.Normal('y_pred', mu=mu, sigma=sigma, observed=data.length)

        # Sample from the posterior to estimate distributions for model parameters.
        idata_vv = pm.sample()
    
    with pm.Model() as model_nv:
        # Define 'month' as shared mutable data for flexibility in posterior predictive sampling.
        x_shared = pm.MutableData('x_shared', data.month.values.astype(float))

        # Priors for the linear model of the mean (μ):
        # α (intercept): Represents the estimated average length at month=0.
        alpha = pm.Normal('alpha', sigma=10)

        # β (slope): Represents the rate of change in mean length as a function of sqrt(month).
        beta = pm.Normal('beta', sigma=10)

        sigma = pm.HalfNormal('sigma', 5)

        # Linear model for mean (μ): μ = α + β * sqrt(month)
        # Using the square root of the predictor to account for potential non-linear relationships.
        mu = pm.Deterministic('mu', alpha + beta * x_shared**0.5)

        # Remove individual sigmas per person
        # Linear model for standard deviation (σ): σ = γ + δ * month
        # Allows σ to vary linearly with month, addressing heteroskedasticity.
        # sigma = pm.Deterministic('sigma', gamma + delta * x_shared)

        # Likelihood function: Observed length is modeled as a normal distribution.
        # The mean and standard deviation are modeled as functions of the predictor variable.
        y_pred = pm.Normal('y_pred', mu=mu, sigma=sigma, observed=data.length)

        # Sample from the posterior to estimate distributions for model parameters.
        idata_nv = pm.sample()

    # Plot the results: posterior mean and uncertainty intervals
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left panel: Posterior predictive mean and intervals
    axes[0].plot(data.month, data.length, "C0.", alpha=0.1, label="Observed data")  # Scatter plot of data.
    
    # Extract posterior samples
    posterior = az.extract(idata_vv)

    # Compute posterior mean estimates for μ (mean) and σ (standard deviation).
    mu_m = posterior["mu"].mean("sample").values  # Mean response.
    sigma_m = posterior["sigma"].mean("sample").values  # Standard deviation of response.

    # Plot the posterior mean of the response variable (μ).
    axes[0].plot(data.month, mu_m, c="k", label="Posterior mean (μ)")

    # Plot 1 standard deviation bands (mean ± σ)
    axes[0].fill_between(data.month, mu_m + 1 * sigma_m, mu_m - 1 * sigma_m,
                         alpha=0.6, color="C1", label="1 SD interval")

    # Plot 2 standard deviation bands (mean ± 2σ)
    axes[0].fill_between(data.month, mu_m + 2 * sigma_m, mu_m - 2 * sigma_m,
                         alpha=0.4, color="C1", label="2 SD interval")

    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Length")
    axes[0].legend()

    # Right panel: Plot estimated standard deviation (σ) as a function of month
    axes[1].plot(data.month, sigma_m, label="Posterior mean (σ)")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel(r"$\bar \sigma$", rotation=0)
    axes[1].legend()

    plt.show()

    # Plot the results: posterior mean and uncertainty intervals
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left panel: Posterior predictive mean and intervals
    axes[0].plot(data.month, data.length, "C0.", alpha=0.1, label="Observed data")  # Scatter plot of data.
    
    # Extract posterior samples
    posterior = az.extract(idata_nv)

    # Compute posterior mean estimates for μ (mean) and σ (standard deviation).
    mu_m = posterior["mu"].mean("sample").values  # Mean response.
    sigma_m = posterior["sigma"].mean("sample").values  # Standard deviation of response.

    # Plot the posterior mean of the response variable (μ).
    axes[0].plot(data.month, mu_m, c="k", label="Posterior mean (μ)")

    # Plot 1 standard deviation bands (mean ± σ)
    axes[0].fill_between(data.month, mu_m + 1 * sigma_m, mu_m - 1 * sigma_m,
                         alpha=0.6, color="C1", label="1 SD interval")

    # Plot 2 standard deviation bands (mean ± 2σ)
    axes[0].fill_between(data.month, mu_m + 2 * sigma_m, mu_m - 2 * sigma_m,
                         alpha=0.4, color="C1", label="2 SD interval")

    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Length")
    axes[0].legend()

    # Right panel: Plot estimated standard deviation (σ) as a function of month
    axes[1].plot(data.month, np.repeat(sigma_m, len(data.month)), label="Posterior mean (σ)")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel(r"$\bar \sigma$", rotation=0)
    axes[1].legend()

    plt.show()

    # Posterior predictive checks (PPC)
    with model_vv:
        # Update the shared data with a new value for posterior predictive sampling.
        # Here, we are predicting for a baby 0.5 months old.
        pm.set_data({'x_shared': [0.5]})

        # Sample from the posterior predictive distribution.
        ppc = pm.sample_posterior_predictive(idata_vv)

        # Extract the predicted response variable.
        y_ppc = ppc.posterior_predictive['y_pred'].stack(sample=('chain', 'draw'))

        # Reference length for percentile calculation.
        ref = 52.5

        # Calculate kernel density estimate for the posterior predictive samples.
        grid, pdf = az.stats.density_utils._kde_linear(y_ppc.values)

        # Plot the posterior predictive distribution.
        plt.plot(grid, pdf, label="Posterior predictive distribution")

        # Calculate and highlight the percentile for the reference length.
        percentile = int((y_ppc <= ref).mean() * 100)
        plt.fill_between(
            grid[grid < ref],
            pdf[grid < ref],
            label=f"Percentile = {percentile}",
            color="C2",
            alpha=0.5,
        )

        plt.xlabel("Length")
        plt.yticks([])
        plt.legend()
    plt.show()

    # Posterior predictive checks (PPC)
    with model_nv:
        # Update the shared data with a new value for posterior predictive sampling.
        # Here, we are predicting for a baby 0.5 months old.
        pm.set_data({'x_shared': [0.5]})

        # Sample from the posterior predictive distribution.
        ppc = pm.sample_posterior_predictive(idata_nv)

        # Extract the predicted response variable.
        y_ppc = ppc.posterior_predictive['y_pred'].stack(sample=('chain', 'draw'))

        # Reference length for percentile calculation.
        ref = 52.5

        # Calculate kernel density estimate for the posterior predictive samples.
        grid, pdf = az.stats.density_utils._kde_linear(y_ppc.values)

        # Plot the posterior predictive distribution.
        plt.plot(grid, pdf, label="Posterior predictive distribution")

        # Calculate and highlight the percentile for the reference length.
        percentile = int((y_ppc <= ref).mean() * 100)
        plt.fill_between(
            grid[grid < ref],
            pdf[grid < ref],
            label=f"Percentile = {percentile}",
            color="C2",
            alpha=0.5,
        )

        plt.xlabel("Length")
        plt.yticks([])
        plt.legend()
    plt.show()
# %% Hierarchical Linear Regression

# Simulate hierarchical regression data
# N: Number of data points per group (except the last group).
# groups: Eight distinct groups (A-H).
# M: Total number of groups.
N = 20
groups = ["A", "B", "C", "D", "E", "F", "G", "H"]
M = len(groups)

# Assign group indices to each data point.
# The last group ("H") has only one data point, demonstrating information sharing in hierarchical models.
idx = np.repeat(range(M - 1), N)
idx = np.append(idx, 7)  # Group H has only one data point.

# Generate true underlying parameters for the simulation.
np.random.seed(314)  # Seed for reproducibility.
alpha_real = np.random.normal(2.5, 0.5, size=M)  # Intercepts for each group.
beta_real = np.random.beta(6, 1, size=M)  # Slopes for each group.
eps_real = np.random.normal(0, 0.5, size=len(idx))  # Random noise (epsilon).

# Generate predictor (x_m) and response (y_m) data.
x_m = np.random.normal(0, 1, len(idx))  # Predictor variable, drawn from standard normal distribution.
y_m = alpha_real[idx] + beta_real[idx] * x_m + eps_real  # Response variable.

# Plot the simulated data for each group.
_, ax = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
ax = np.ravel(ax)
j, k = 0, N
for i, g in enumerate(groups):
    # Scatter plot of data points for each group.
    ax[i].scatter(x_m[j:k], y_m[j:k], marker=".")
    ax[i].set_title(f"group {g}")
    j += N
    k += N
plt.show()

plt.scatter(x_m,  y_m)
plt.title('Data Without Groups')
plt.show()

# Coordinate definitions for PyMC models.
coords = {'group': groups}

# Non-hierarchical (unpooled) model
if __name__ == '__main__':
    with pm.Model(coords=coords) as unpooled_model:
        # Priors for group-specific intercepts and slopes.
        # Each group gets its own independent alpha (intercept) and beta (slope).
        alpha = pm.Normal('alpha', mu=0, sigma=10, dims='group')
        beta = pm.Normal('beta', mu=0, sigma=10, dims='group')
        sigma = pm.HalfNormal('sigma', 5)  # Shared noise variance.

        # Likelihood: Linear regression for each group's data.
        _ = pm.Normal('y_pred', mu=alpha[idx] + beta[idx] * x_m, sigma=sigma, observed=y_m)

        # Sample posterior distributions for parameters.
        idata_up = pm.sample(random_seed=123)

    # Visualize posterior distributions for alpha and beta.
    az.plot_forest(idata_up, var_names=["alpha", "beta"], combined=True, figsize=(10, 4))
    plt.show()

# Hierarchical centered model
if __name__ == '__main__':
    with pm.Model(coords=coords) as hierarchical_centered:
        # Hyperpriors for group-level parameters (shared across groups).
        alpha_mu = pm.Normal('alpha_mu', mu=y_m.mean(), sigma=1)  # Mean of group intercepts.
        alpha_sigma = pm.HalfNormal('alpha_sigma', 5)  # SD of group intercepts.
        beta_mu = pm.Normal('beta_mu', mu=0, sigma=1)  # Mean of group slopes.
        beta_sigma = pm.HalfNormal('beta_sigma', 5)  # SD of group slopes.

        # Group-level priors: Individual group parameters are drawn from hyperpriors.
        alpha = pm.Normal('alpha', mu=alpha_mu, sigma=alpha_sigma, dims='group')
        beta = pm.Normal('beta', mu=beta_mu, sigma=beta_sigma, dims='group')
        sigma = pm.HalfNormal('sigma', 5)  # Shared noise variance.

        # Likelihood: Linear regression with group-specific parameters.
        _ = pm.Normal('y_pred', mu=alpha[idx] + beta[idx] * x_m, sigma=sigma, observed=y_m)

        # Sample posterior distributions.
        idata_cen = pm.sample(random_seed=123)

    # Visualize posterior distributions for alpha and beta.
    az.plot_forest(idata_cen, var_names=["alpha", "beta"], combined=True, figsize=(10, 4))
    plt.show()

    # Hierarchical non-centered model
    with pm.Model(coords=coords) as hierarchical_non_centered:
        # Hyperpriors for group-level parameters.
        alpha_mu = pm.Normal('alpha_mu', mu=y_m.mean(), sigma=1)  # Mean of group intercepts.
        alpha_sigma = pm.HalfNormal('alpha_sigma', 5)  # SD of group intercepts.
        beta_mu = pm.Normal('beta_mu', mu=0, sigma=1)  # Mean of group slopes.
        beta_sigma = pm.HalfNormal('beta_sigma', 5)  # SD of group slopes.

        # Non-centered parameterization for slopes.
        alpha = pm.Normal('alpha', mu=alpha_mu, sigma=alpha_sigma, dims='group')
        beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, dims='group')
        beta = pm.Deterministic('beta', beta_mu + beta_offset * beta_sigma, dims='group')

        sigma = pm.HalfNormal('sigma', 5)  # Shared noise variance.

        # Likelihood: Linear regression with group-specific parameters.
        _ = pm.Normal('y_pred', mu=alpha[idx] + beta[idx] * x_m, sigma=sigma, observed=y_m)

        # Sample posterior distributions.
        idata_ncen = pm.sample(random_seed=123)

    # Visualize posterior distributions for alpha and beta.
    az.plot_forest(idata_ncen, var_names=["alpha", "beta"], combined=True, figsize=(10, 4))

    # Compare centered and non-centered models.
    az.plot_forest([idata_cen, idata_ncen], var_names=["alpha", "beta"], combined=True, figsize=(10, 4))
    plt.show()

    # Plot fitted lines for each group using the non-centered model.
    _, ax = plt.subplots(2, 4, figsize=(12, 5), sharex=True, sharey=True)
    ax = np.ravel(ax)
    j, k = 0, N
    x_range = np.linspace(x_m.min(), x_m.max(), 10)

    # Extract posterior samples for group-level parameters.
    posterior = az.extract(idata_ncen)

    for i, g in enumerate(groups):
        ax[i].scatter(x_m[j:k], y_m[j:k], marker=".")  # Data points for each group.
        ax[i].set_xlabel(f"$x_{i}$")  # Predictor axis label.
        ax[i].set_ylabel(f"$y_{i}$", labelpad=10, rotation=0)  # Response axis label.

        # Compute mean posterior estimates for alpha and beta for the group.
        alfas = posterior["alpha"].sel(group=g)
        betas = posterior["beta"].sel(group=g)
        alfa_m = alfas.mean("sample").item()
        beta_m = betas.mean("sample").item()

        # Plot mean regression line for the group.
        ax[i].plot(x_range, alfa_m + beta_m * x_range, c="k")

        # Plot the HDI (highest density interval) for the regression line.
        az.plot_hdi(x_range, alfas + betas * xr.DataArray(x_range).transpose(), ax=ax[i])

        plt.xlim(x_m.min() - 1, x_m.max() + 1)
        plt.ylim(y_m.min() - 1, y_m.max() + 1)

        j += N
        k += N
    plt.show()

# %%
