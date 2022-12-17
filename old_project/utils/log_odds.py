"""
Actually incorporate the multinomial model
"""
import numpy as np
from regression import get_X
from mcmc_samplers import beta_gibbs_sampling
from modeling import hammer_and_pickle, ols_regressors

def link_function(response_array: np.array):
    """
    Link function for connecting predicted data to physical observables
    """
    post_odds_total = sum([np.exp(eta) for eta in response_array])
    return np.array([np.exp(eta) / post_odds_total for eta in response_array])



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    spec_year = 1982
    years = [i for i in range(1982, 2019) if not i in [1993, 1994, 1995, 1996]]
    X = np.array([get_X(year) for year in years[::2]]) # X[0] is 50 x 3
    Y = hammer_and_pickle() # len(Y) = 17 (years) // len(Y[0]) = 50 (states) // len(Y[0][0]) = 4 (parties) // len(Y[0][0][0]) = int
    beta_ols = ols_regressors(X, Y) # len(beta_ols) = 17 // len(beta_ols[0]) = 3 // len(beta_ols[0][0]) = int
    # print(len(Y[0][0])) 
    mcmc_results = []
    for p in ["democrat", "republican", "libertarian", "other"]:
        beta_post, _ = beta_gibbs_sampling(spec_year, X, Y, beta_ols, p, 1000, 500)
        mcmc_beta = np.array([
            np.mean([b[0] for b in beta_post]), np.mean([b[1] for b in beta_post]), np.mean([b[2] for b in beta_post])
            ])
        mcmc_results.append(mcmc_beta)
    # print(f"b1: {beta1}\nb2: {beta2}\nb3: {beta3}")
    mcmcY = np.matmul(X[0], mcmc_results[0]) # predicted data at a particular year
    mcmcY_r = np.matmul(X[0], mcmc_results[1]) # predicted data at a particular year
    mcmcY_l = np.matmul(X[0], mcmc_results[2]) # predicted data at a particular year
    mcmcY_o = np.matmul(X[0], mcmc_results[3]) # predicted data at a particular year
    # mcmcY = np.matmul(X[1], mcmc_results[0]) # predicted data at a particular year
    # mcmcY = np.matmul(X[2], mcmc_results[0]) # predicted data at a particular year
    # mcmcY = np.matmul(X[3], mcmc_results[0]) # predicted data at a particular year
    spec_total_votes = sum([y[0] for y in Y[0]])
    observ_Y = np.array([y[0] / spec_total_votes for y in Y[0]])
    markov_Y = link_function(mcmcY)
    spec_total_votes = sum([y[1] for y in Y[0]])
    observ_Y = np.array([y[0] / spec_total_votes for y in Y[0]])
    markov_Y_r = link_function(mcmcY_r)
    spec_total_votes = sum([y[1] for y in Y[0]])
    observ_Y = np.array([y[1] / spec_total_votes for y in Y[0]])
    markov_Y_l = link_function(mcmcY_l)
    spec_total_votes = sum([y[2] for y in Y[0]])
    observ_Y = np.array([y[2] / spec_total_votes for y in Y[0]])
    markov_Y_o = link_function(mcmcY_o)
    spec_total_votes = sum([y[3] for y in Y[0]])
    observ_Y = np.array([y[3] / spec_total_votes for y in Y[0]])
    plt.figure(figsize=(8,6))
    plt.scatter(observ_Y, markov_Y, c="c", label="Democrat")
    plt.scatter(observ_Y, markov_Y_r, c="m", label="Republican")
    plt.scatter(observ_Y, markov_Y_l, c="y", label="Libertarian")
    plt.scatter(observ_Y, markov_Y_o, c="k", label="Other")
    plt.xlabel("Observed"); plt.ylabel("Predicted")
    plt.xlim([-1*10**(-10), 1*10**(-10)])
    plt.legend()
    plt.show()