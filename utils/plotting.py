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
    from modeling import ols_regressors
    years = [i for i in range(1981, 2019) if not i in [1993, 1994, 1995, 1996]]
    X = np.array([get_X(year) for year in years]) # X[0] is 50 x 3
    Y = np.array(pickle.load(open("observed.pkl", "rb"))) # Y is 34 x 50
    beta_ols = ols_regressors(X, Y) # beta_ols is 34 x 3
    beta_post, gamma_post = beta_gibbs_sampling(1982, X, beta_ols, 5000, 100)
    mcmc_beta = np.array([
        np.mean([b[0] for b in beta_post]), np.mean([b[1] for b in beta_post]), np.mean([b[2] for b in beta_post])
        ])
    print(beta_ols[1982 % 1981])
    print(mcmc_beta)

    # Plotting
    plt.figure(figsize=(8,6))
    mcmc_Y = np.matmul(X[1], mcmc_beta)
    plt.scatter(Y[1], mcmc_Y, c="r", s=40)
    plt.xlabel("Observed Votes"); plt.ylabel("Predicted Votes")
    plt.show()
