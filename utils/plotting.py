"""
Script for plotting results
"""
import pickle 
import numpy as np
from itertools import tee
from regression import get_X
import matplotlib.pyplot as plt
from mcmc_samplers import beta_gibbs_sampling

if __name__ == "__main__":
    from modeling import hammer_and_pickle, ols_regressors
    from mcmc_samplers import beta_gibbs_sampling
    years = [i for i in range(1982, 2019) if not i in [1993, 1994, 1995, 1996]]
    X = np.array([get_X(year) for year in years[::2]]) # X[0] is 50 x 3
    Y = hammer_and_pickle() # len(Y) = 17 (years) // len(Y[0]) = 50 (states) // len(Y[0][0]) = 4 (parties) // len(Y[0][0][0]) = int
    beta_ols = ols_regressors(X, Y) # len(beta_ols) = 17 // len(beta_ols[0]) = 3 // len(beta_ols[0][0]) = int
    # print(len(Y[0][0])) 
    beta_post, gamma_post = beta_gibbs_sampling(1982, X, Y, beta_ols, 5000, 1000)
    mcmc_beta = np.array([
        np.mean([b[0] for b in beta_post]), np.mean([b[1] for b in beta_post]), np.mean([b[2] for b in beta_post])
        ])
    # print(f"b1: {beta1}\nb2: {beta2}\nb3: {beta3}")
    mcmcY = np.matmul(X[0], mcmc_beta)
    # Plotting
    plt.figure(figsize=(8,6))
    plt.plot(Y, mcmcY)
