"""
...

Authors
Sam Dawley & Oliver Wolff
"""
import pickle
import numpy as np
import pandas as pd
from numpy.linalg import inv
from regression import get_X
from modeling import hammer_and_pickle
from scipy.stats import multivariate_normal, gamma

def beta_gibbs_sampling(year: int, covariates: np.array, observed: np.array, beta_ols: np.array, N: int, burn: int) -> np.array:
    """
    Gibbs sampling algorithm to get estimate of the regression coefficients
    """
    # Global constants
    time = year % 1981
    Y = observed # FIGURE OUT FILE PATHS
    beta_ols, X, Y = beta_ols[time], covariates[time], Y[time].T # DETERMINE BETTER WAY TO SLICE COVARIATES PER YEAR
    ssr_beta = np.matmul( (Y.T - np.matmul(X, beta_ols)).T, (Y.T - np.matmul(X, beta_ols)) ) # ssr_beta is 34 x 34
    ssr_beta = np.mean([[np.mean(a) for a in l ] for l in ssr_beta])
    n = len(X)
    nu0, var0 = 1, ssr_beta / (n - len(beta_ols)) # Hyperparameters for precision

    # Prior paramaters/initial values
    prior_beta_mean, prior_beta_cov = beta_ols[0][1:], inv(np.matmul(X.T, X))/(n*var0)
    prior_gamma_loc, prior_gamma_scale = nu0/2, 2/(nu0*var0)
    # manipulate vector of votes
    prior_beta = multivariate_normal(mean=prior_beta_mean, cov=prior_beta_cov).rvs()
    prior_gamma = gamma(a=prior_gamma_loc, scale=prior_gamma_scale).rvs()

    # Posterior arrays
    beta_posterior, gamma_posterior = np.zeros((N-burn, 3)), np.zeros(N-burn)
    for t in range(N):
        print(f"GIBBS SAMPLER ITERATION {t+1}")
        # Update parameters
        # For beta
        beta_cov = inv(inv(prior_beta_cov) + prior_gamma*np.matmul(X.T, X))
        beta_mean = np.matmul(beta_cov, np.matmul(prior_beta_cov, beta_ols) + prior_gamma*np.matmul(X.T, Y.T))[0][1:]
        vbeta = multivariate_normal(mean=beta_mean, cov=beta_cov).rvs()
        # For gamma
        new_ssr_beta = np.matmul( (Y - np.matmul(X, vbeta)).T, (Y - np.matmul(X, vbeta)) )
        prior_gamma_loc, prior_gamma_scale = (nu0 + n)/2, 2/(nu0*var0 + new_ssr_beta)
        prior_gamma = gamma(a=prior_gamma_loc, scale=prior_gamma_scale).rvs()[0][0]
        # If we're past the burn-in period, append realizations of the parameters to posterior arrays
        if t >= burn:
            # print(f"Beta estimate : {vbeta}")
            beta_posterior[t-burn] += vbeta
            gamma_posterior[t-burn] += prior_gamma
    return beta_posterior, gamma_posterior


if __name__ == "__main__":
    from modeling import ols_regressors
    years = [i for i in range(1982, 2019) if not i in [1993, 1994, 1995, 1996]]
    X = np.array([get_X(year) for year in years[::2]]) # X[0] is 50 x 3
    Y = hammer_and_pickle() # len(Y) = 17 (years) // len(Y[0]) = 50 (states) // len(Y[0][0]) = 4 (parties) // len(Y[0][0][0]) = int
    beta_ols = ols_regressors(X, Y) # len(beta_ols) = 17 // len(beta_ols[0]) = 3 // len(beta_ols[0][0]) = int
    # print(len(Y[0][0])) 
    beta_post, gamma_post = beta_gibbs_sampling(1982, X, Y, beta_ols, 2000, 1000)
    mcmcY = 1
    # Plotting
    plt.figure(figsize=(8,6))
    plt.plot()



