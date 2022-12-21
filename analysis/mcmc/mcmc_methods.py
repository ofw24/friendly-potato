"""
Script for extracting and manipulating data
"""
import numpy as np
import pandas as pd
from functools import partial
from scipy.stats import invgamma, norm ,truncnorm, uniform
from analysis.data_manips.data_extraction import ale
from analysis.mcmc.not_mcmc_methods import double_logistic

def gibbs_sampler(data: pd.DataFrame, country: str, beta_ols: np.array, N: int, burn: int) -> np.array:
    """
    Metropolis-Hastings sampling algorithm
    Ordering for all length-6 arrays is : Delta_1, Delta_2, Delta_3, Delta_4, k, z
    """
    # Global data
    actual_life_exps = ale(data, country).flatten()
    n = len(actual_life_exps)
    nu0 = 4
    nun = nu0 + n
    kappa0 = 1
    sample_var = np.var(actual_life_exps)
    ybar = np.mean(actual_life_exps)

    # Initialize prior variances
    rate_params = np.array([15.6**2, 23.5**2, 14.5**2, 14.7**2, 3.5**2, 0.6**2])
    variai = np.array([invgamma(a=nu0/2, scale=rate).rvs() for rate in rate_params]) # Variances for all of the 6 parameters
    aS, dS = np.array([15.77, 40.97, 0.21, 19.82, 2.93, 0.40]), np.array([15.77**2, 40.97**2, 0.21**2, 19.82**2, 2.93**2, 0.40**2])
    lbS, ubS = np.array([0, 0, 0, 0, 0, 0]), np.array([100, 100, 100, 100, 10, 1.15]) 
    prior_param_rvs = np.array([truncnorm(a=lb, b=ub, loc=a, scale=d).rvs() for a, d, lb, ub in zip(aS, dS, lbS, ubS)])
    # prior_param_rvs = [b for b in beta_ols]

    # Initialize array to store results
    posterior_params = []
    
    # Start the sampling
    for t in range(N):
        # print(f"METROPOLIS-HASTINGS ITERATION {t+1}")
        u = uniform().rvs()
        # Update parameters for
        # Variai
        new_variai = []
        for parameter, mean, var, prior_rate, life_exp in zip(prior_param_rvs, aS, variai, rate_params, actual_life_exps):
            likelihood = double_logistic(prior_param_rvs, life_exp)
            tmp_var = ((nu0*var + (n-1)*sample_var) + kappa0*n*(ybar-mean)/(kappa0 + n)) / nun
            proposed_point = invgamma(a=nun/2, scale=1/((nu0+n)*tmp_var/2)).rvs()
            proposed_prob  = invgamma(a=nun/2, scale=1/((nu0+n)*tmp_var/2)).pdf(proposed_point)
            prior_prob = invgamma(a=nu0/2, scale=1/prior_rate).pdf(var)
            # print(prior_prob)
            #
            acceptance_ratio = (proposed_prob / prior_prob) / 10**3 # Scaling because of non-normalization
            if min(np.abs(acceptance_ratio), 1) >= u:
                new_variai.append(proposed_point)
            else:
                new_variai.append(var)
        # print(f"Prior      {prior_prob}")
        # print(f"Posterior  {proposed_prob}")
        variai = []; variai = np.array(new_variai)
        testing = acceptance_ratio

        # Delta_i's
        param_rvs = []
        for parameter, var, lb, ub, a, d, life_exp in zip(prior_param_rvs, variai, lbS, ubS, aS, dS, actual_life_exps):
            likelihood = double_logistic(prior_param_rvs, life_exp)
            proposed_point = truncnorm(a=lb, b=ub, loc=parameter, scale=var).rvs()
            proposed_prob  = truncnorm(a=lb, b=ub, loc=parameter, scale=var).pdf(proposed_point) * likelihood
            prior_prob = truncnorm(a=lb, b=ub, loc=a, scale=d).pdf(parameter) 
            #
            acceptance_ratio = (proposed_prob / prior_prob) / 10**5 # Scaling because of non-normalization
            if min(np.abs(acceptance_ratio), 1) >= u:
                param_rvs.append(proposed_point)
            else:
                param_rvs.append(parameter)
        prior_param_rvs = []; prior_param_rvs = np.array(param_rvs)

        if t > burn:
            posterior_params.append(prior_param_rvs)

    print(f"{u} vs. {min(np.abs(acceptance_ratio), 1)} vs {min(np.abs(acceptance_ratio), 1)}")
    return np.array(posterior_params)

if __name__ == "__main__":
    pass