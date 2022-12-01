"""
Script for storing various Markov Chain Monte Carlo sampling methods
Many of these are case-specific and need to be modified in the future for more general use

Authors
Sam Dawley & Oliver Wolff
"""
import numpy as np
from numpy.linalg import inv, norm
from math import factorial
import pandas as pd
from scipy.stats import multivariate_normal, invgamma, rv_continuous

def infinisum(f: callable):
    n, res = 0, f(0)
    while True:
        term = sum( f(k) for k in range(2**n,2**(n+1)) )
        if (res+term)-res == 0:
             break;
        n,res = n+1, res+term
    return res

class CMP(rv_continuous):
    """
    Conway-Maxwell Poisson (CMP) probability distribution
    """
    def __init__(self, loc: float, scale: float, a: int=0, b: int=np.infty):
        if loc <= 0 or scale <= 0:
            raise ValueError("Location and scale parameters of CMP distribution must be greater than zero.")
        self.loc = loc
        self.scale = scale
    def pdf(self, y: float) -> float:
        """
        Probability mass function of the CMP distribution
        Returns probability of observing y
        """
        Q = np.sum([self.loc**j / (factorial(j) ** self.scale) for j in range(0, int(np.infty))])
        f = (self.loc ** y) / (factorial(y) ** self.scale) / Q
        return f / Q
    def rvs(self, N: int=1) -> np.array:
        """
        Draw N random variates from the distribution
        Algorithm for generating random variates attributed to 
            https://rossetti.github.io/RossettiArenaBook/app-rnrv-rvs.html
        Parameters
        ----------
        N (int) is the number of random variates generated

        Returns
        -------
        instances (np.array) contains all random variates
        """
        instances = np.zeros(N)
        for t in range(N):
            instances[t] += np.sqrt( -np.log(1- np.random.rand() )  / self.theta)
        return instances


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


if __name__ == "__main__":
    a = CMP(0.5, 1)
    # a.pdf(5)
    # a = integers(blk_size=1)
    ggg = lambda x: x+1
    converge_sum(ggg, integers(blk_size=5))