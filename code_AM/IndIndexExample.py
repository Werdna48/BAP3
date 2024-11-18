#%% Import necessary libraries
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
az.style.use("arviz-grayscale")
from cycler import cycler
default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])

#%% Step 1: Simulate Data
# Set random seed for reproducibility
np.random.seed(123)

# Define participants and number of observations
participants = ["P1", "P2", "P3", "P4", "P5"]
n_observations = 50  # Observations per participant

# True group-level parameters
true_mu_hyp = 5      # True group-level mean
true_sigma_hyp = 1.5 # True group-level standard deviation for participant means

# True participant-specific parameters
true_mu = np.random.normal(true_mu_hyp, true_sigma_hyp, size=len(participants))  # Means for each participant
true_sigma = np.random.uniform(1, 1.5, size=len(participants))  # Standard deviations for each participant

# Generate observations (y) for each participant
simulated_data = {
    "participant": np.repeat(participants, n_observations),
    "y": np.concatenate([np.random.normal(mu, sigma, n_observations) for mu, sigma in zip(true_mu, true_sigma)]),
}

# Convert simulated data to a DataFrame
data = pd.DataFrame(simulated_data)

# Map participant labels to indices for indexing in the model
participant_idx = pd.Categorical(data["participant"], categories=participants).codes

# Define coordinates for PyMC model
coords = {"participants": participants, "observations": data.index}

#%% Step 2: Build Hierarchical Bayesian Model
if __name__ == "__main__":
    with pm.Model(coords=coords) as participant_model:
        # Hyperpriors (group-level parameters)
        mu_hyp = pm.Normal("mu_hyp", mu=5, sigma=2)          # Group-level mean
        sigma_hyp = pm.HalfNormal("sigma_hyp", sigma=2)      # Group-level variability

        # Priors for participant-level parameters
        mu = pm.Normal("mu", mu=mu_hyp, sigma=sigma_hyp, dims="participants")  # Means for participants
        sigma = pm.HalfNormal("sigma", sigma=5, dims="participants")           # SDs for participants

        # Likelihood for observations
        y = pm.Normal("y", mu=mu[participant_idx], sigma=sigma[participant_idx], 
                      observed=data["y"], dims="observations")
        
        # Sample from the posterior
        idata = pm.sample(random_seed=42)

        # Sample from the posterior predictive
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=42))

#%% Step 3: Visualize Posterior Results
# Plot trace plots for posterior distributions
az.plot_trace(idata, var_names=["mu_hyp", "sigma_hyp", "mu", "sigma"])
plt.show()

# Summarize posterior estimates
summary = az.summary(idata, var_names=["mu_hyp", "sigma_hyp", "mu", "sigma"], round_to=2)
print(summary)

#%% Step 4: Compare Posterior Estimates to True Parameters in Participant Row Format
# Extract posterior estimates for participant-specific parameters
posterior_mu = idata.posterior["mu"].mean(dim=["chain", "draw"]).values
posterior_sigma = idata.posterior["sigma"].mean(dim=["chain", "draw"]).values

# Extract posterior estimates for group-level parameters
posterior_mu_hyp = idata.posterior["mu_hyp"].mean(dim=["chain", "draw"]).item()
posterior_sigma_hyp = idata.posterior["sigma_hyp"].mean(dim=["chain", "draw"]).item()

# Build a row-structured table
rows = [
    {
        "Participant": "Group",
        "True Mu": true_mu_hyp,
        "Posterior Mu": posterior_mu_hyp,
        "True Sigma": true_sigma_hyp,
        "Posterior Sigma": posterior_sigma_hyp,
    }
]

# Add rows for each participant dynamically
for i, p in enumerate(participants):
    rows.append({
        "Participant": p,
        "True Mu": true_mu[i],
        "Posterior Mu": posterior_mu[i],
        "True Sigma": true_sigma[i],
        "Posterior Sigma": posterior_sigma[i],
    })

# Convert rows into a DataFrame for visualization
comparison_table_participants = pd.DataFrame(rows)

#%% Step 5: Posterior Predictive Checks
# Visualize posterior predictive checks to assess model fit
az.plot_ppc(idata, num_pp_samples=100)
plt.show()

# %%