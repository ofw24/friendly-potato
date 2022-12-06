"""
...

Authors
Sam Dawley & Oliver Wolff
"""
import numpy as np
import pandas as pd
from numpy.linalg import inv
from dist_classes import CMP_posterior
from scipy.stats import multivariate_normal, invgamma, rv_continuous, norm, uniform


def latent_beta_distribution(param: np.array, var: float, X: np.array, Psi: np.array) -> multivariate_normal:
    """
    Multivariate normal distribution of beta_lambda
    Returns an instance of the multivariate_normal distribution

    loc = lambda
    sigma = variance
    X = matrix of covariates
    Psi = matrix of spatial basis vectors
    """
    Px = np.matmul( X, np.matmul( inv(np.matmul( X.T, X )), X.T ) )
    Ps = np.matmul( Psi, np.matmul( inv(np.matmul( Psi.T, Psi )), Psi.T ) )
    loc = np.matmul((Px + Ps - np.matmul(Ps, Px)), np.log(param))
    covariance = var*np.ones(len(param))
    return multivariate_normal(mean=loc, cov=covariance)
def variance_distribution(beta: np.array, mean: np.array) -> invgamma:
    """
    Inverse Gamma distribution of the latent variance
    Returns an instance of the invgamma distribution

    beta = vector of transformed latent variables
    mean = mean of latent variable distribution
    """
    shape = 1 + len(beta)/2
    over_scale = np.matmul((beta-mean).T, (beta-mean)) / 2
    return invgamma(a=shape, scale=1/over_scale)
# def joint_latent_posterior_predictive(
#     param: np.array, var: float, X: np.array, Psi: np.array, Y: np.array, a: float, b: float, w: float, N: int, burn: int
#     ) -> tuple:
#     """
#     Joint posterior predictive distribution of (beta_lambda, beta_nu)
#     """
#     # Initialize parameters which are unchanging with each iteration
#     Px = np.matmul( X, np.matmul( inv(np.matmul( X.T, X )), X.T ) )
#     Ps = np.matmul( Psi, np.matmul( inv(np.matmul( Psi.T, Psi )), Psi.T ) )
#     # Initialize parameters to begin the Gibbs Sampler
#     var_lambda, var_nu = invgamma(a=1, scale=1).rvs(), invgamma(a=1, scale=1).rvs()
#     for t in range(N):
#         lambda_nu_posterior(loc: float, scale: float, y: int, a: float, b: float, w: float)

#     return


# # Gibbs sampling algorithm
# def gibbs_sampler(
#     t1_conditional: norm,
#     t2_conditional: norm,
#     N: int, burn: int
#     ) -> tuple:
#     """
#     ...

#     Parameters
#     ----------
#     t1_conditional (callable) full conditional posterior distribution of theta_1
#     t2_conditional (callable) full conditional posterior distribution of theta_2
#     N (int) is the number of iterations for the sampler
#     burn (int) is the burn-in period

#     Returns 
#     -------
#     t1_posterior (np.array) is the empirical posterior distribution of theta_1
#     t2_posterior (np.array) is the empirical posterior distribution of theta_2
#     """
#     # Global variables from given data
#     global a1, a2, b1, b2, sample_data, n
#     # Create arrays for storing posterior distributions of parameters
#     t1_posterior, t2_posterior = np.zeros(N-burn), np.zeros(N-burn)
#     # Create prior parameter for initialization
#     t1_prior, t2_prior = 25, 25
#     # Generate some parameters which are unchanging with each iteration
#     ensemble_sum = sum(sample_data)
#     t1_scale, t2_scale = 1 / (1/b1**2 + n), 1 / (1/b2**2 + n)
#     t1_partial_loc, t2_partial_loc = a1/b1**2 + ensemble_sum, a2/b2**2 + ensemble_sum
#     # Update parameters from full conditional distributions and append to posterior arrays
#     for t in range(N):
#         print(f"GIBBS SAMPLER ITERATION {t+1}")
#         # Generate posterior parameters and update parameters
#         # For theta_1
#         t1_loc = (t1_partial_loc - n*t2_prior) / t1_scale
#         t1_post = t1_conditional(loc=t1_loc, scale=t1_scale).rvs()
#         test = np.random.rand()
#         # test = 1
#         if test > 0.7: # Implement random updates (quasi-metropolis-hastings, if you will)
#             t1_prior = t1_post
#         # For theta_2
#         t2_loc = (t2_partial_loc - n*t1_prior) / t2_scale
#         t2_post = t2_conditional(loc=t2_loc, scale=t2_scale).rvs()
#         if test > 0.7:
#             t2_prior = t2_post
#         # If we're past the burn-in period, append realizations of the parameters to posterior arrays
#         if t >= burn:
#             t1_posterior[t-burn] += t1_post
#             t2_posterior[t-burn] += t2_post
#     return t1_posterior, t2_posterior



    # X1 = np.linspace(0, 1, 10)
    # X2 = np.linspace(0, 1, 10)
    # Y = d.pdf(loc=X1, scale=X2)
    # print(Y)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(X1, Y)
    # plt.show()


