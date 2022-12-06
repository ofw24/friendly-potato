"""
...

Authors
Sam Dawley & Oliver Wolff
"""
import numpy as np
import pandas as pd
from numpy.linalg import inv
from dist_classes import CMP, CMP_posterior
from scipy.stats import multivariate_normal, invgamma, rv_continuous, norm, uniform


def latent_beta_distribution(param: np.array, var: float, X: np.array, Psi: np.array) -> multivariate_normal:
    """
    Multivariate normal distribution of beta_lambda
    Returns an instance of the multivariate_normal distribution
    P
    -

    """
    Px = np.matmul( X, np.matmul( inv(np.matmul( X.T, X )), X.T ) )
    Ps = np.matmul( Psi, np.matmul( inv(np.matmul( Psi.T, Psi )), Psi.T ) )
    loc = np.matmul((Px + Ps - np.matmul(Ps, Px)), np.log(param))
    covariance = var*np.ones(len(param))
    return multivariate_normal(mean=loc, cov=covariance)
def variance_distribution(beta: np.array, X: np.array, Psi: np.array) -> invgamma:
    """
    Inverse Gamma distribution of the latent variance
    Returns an instance of the invgamma distribution

    beta = vector of transformed latent variables
    mean = mean of latent variable distribution
    """
    shape = 1 + len(beta)/2
    Px = np.matmul( X, np.matmul( inv(np.matmul( X.T, X )), X.T ) )
    Ps = np.matmul( Psi, np.matmul( inv(np.matmul( Psi.T, Psi )), Psi.T ) )
    mean = np.matmul((Px + Ps - np.matmul(Ps, Px)), np.log(np.abs(beta)))
    over_scale = np.matmul((beta-mean).T, (beta-mean)) / 2
    return invgamma(a=shape, scale=1/over_scale)
    
if __name__ == "__main__":
    B = 2
    obs = 10344
    # np.random.seed(3247234)
    x = np.array([[134500, 433454], [234423, 564583165]])
    p = np.array([[47864840, 125], [672732, 218472723]])
    locs, scales = CMP_posterior(100, 0.1, 0.1, 0.1).rvs(initial_loc=2, initial_scale=2, N=B)
    blambda, bnu = np.array([4, 6]), np.array([1, 2])
    for b in range(B):
        beta_lambda = latent_beta_distribution(locs, 1, x, p).rvs()
        beta_nu = latent_beta_distribution(scales, 1, x, p).rvs()
        sigma_lambda = variance_distribution(beta_lambda, x, p).rvs()
        sigma_nu = variance_distribution(beta_nu, x, p).rvs()
        Z = np.zeros(B)
        for i, l, s in zip(range(len(beta_lambda)), beta_lambda, beta_nu):
            Z[i] += CMP(loc=np.exp(l), scale=np.exp(s)).log_pdf(obs)
        print(Z)
    # print(locs)


