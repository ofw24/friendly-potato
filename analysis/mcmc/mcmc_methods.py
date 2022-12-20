"""
Script for extracting and manipulating data
"""
import numpy as np
import pandas as pd
from scipy.stats import invgamma, norm, uniform
from not_mcmc_methods import data_spline

def gibbs_sampler(data: pd.DataFrame, life_exp: np.array, beta_ols: np.array, N: int, burn: int) -> np.array:
    """
    Gibbs sampling algorithm
    Ordering for all length-6 arrays is : Delta_1, Delta_2, Delta_3, Delta_4, k, z
    """
    # Initialize prior variances
    rate_params = np.array([15.6**2, 23.5**2, 14.5**2, 14.7**2, 3.5**2, 0.6**2])
    variai = np.array([invgamma(a=2, scale=1/rate).rvs() for rate in rate_params]) # Variances for all of the 6 parameters
    omega = uniform(0, 10).rvs()
    aS, dS = np.array([15.77, 40.97, 0.21, 19.82, 2.93, 0.40]), np.array([15.77**2, 40.97**2, (1+0.21)**2, 19.82**2, 2.93**2, (1+0.40)**2])
    prior_param_rvs = np.array([norm(a, d).rvs() for a, d in zip(aS, dS)])

    # Initialize array to store results
    posterior_params = []
    
    # Start the sampling
    for t in range(N):
        print(f"GIBBS SAMPLER ITERATION {t+1}")
        # Update parameters for
        # Delta_i's
        param_rvs = []
        for parameter, var in zip(prior_param_rvs, variai):
            param_rvs.append( norm(loc=parameter, scale=var).rvs() )

        prior_param_rvs = np.array(param_rvs)

        if t > burn:
            posterior_params.append(prior_param_rvs)

    return np.array(posterior_params)

if __name__ == "__main__":
    pass