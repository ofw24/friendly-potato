"""
Script for extracting and manipulating data
"""
import numpy as np
import pandas as pd
from scipy.stats import invgamma, norm
from analysis.data_manips.data_extraction import ale

def gibbs_sampler(data: pd.DataFrame, country: str, beta_ols: np.array, N: int, burn: int) -> np.array:
    """
    Gibbs sampling algorithm
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
    variai = np.array([invgamma(a=nu0/2, scale=1/rate).rvs() for rate in rate_params]) # Variances for all of the 6 parameters
    aS, dS = np.array([15.77, 40.97, 0.21, 19.82, 2.93, 0.40]), np.array([15.77**2, 40.97**2, (1+0.21)**2, 19.82**2, 2.93**2, (1+0.40)**2])
    prior_param_rvs = np.array([norm(a, d).rvs() for a, d in zip(aS, dS)])
    # prior_param_rvs = [b for b in beta_ols]

    # Initialize array to store results
    posterior_params = []
    
    # Start the sampling
    for t in range(N):
        # print(f"GIBBS SAMPLER ITERATION {t+1}")
        # Update parameters for
        # Variai
        var_rvs = []
        for parameter, mean, var in zip(prior_param_rvs, aS, variai):
            tmp_var = ((nu0*var + (n-1)*sample_var) + kappa0*n*(ybar-mean)/(kappa0 + n)) / nun
            var_rvs.append( invgamma(a=nun/2, scale=1/((nu0+n)*tmp_var/2)).rvs() )
        variai = np.array(var_rvs)

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