"""
...

Authors
Sam Dawley & Oliver Wolff
"""
import numpy as np
import pandas as pd
from math import factorial
from numpy.linalg import inv
from functools import partial
from scipy.stats import multivariate_normal, invgamma, rv_continuous, norm, uniform

class CMP_posterior(rv_continuous):
    """
    Joint posterior distribution for (lambda, nu) which is conjugate for the Conway-Maxwell Poisson distribution
    """
    def __init__(self, y: int, a: float, b: float, w: float):
        if w <= 0:
            raise ValueError("Joint posterior parameters for (lambda, nu) result in intractable solutions.")
        self.y = y # observed sample point
        self.a = a
        self.b = b
        self.w = w # 0 < w < infty
    def _factorial_logarithm(self):
        try:
            return np.log(float(factorial(self.y)))
        except OverflowError: # Use Stirling's approximation in the case that y! is intractable
            return self.y*np.log(self.y) - self.y
    def pdf(self, loc: float, scale: float) -> float:
        """
        Joint probability density
        Returns probability of observing (lambda, nu)
        Note that lambda = loc, nu = scale
        """
        numerator = (self.y+self.a) * (self.b+self._factorial_logarithm()) * loc**(self.y+self.a-1) * np.exp(-scale*(self.b+self._factorial_logarithm()))
        return numerator / (self.w**(self.y+self.a))
    def _stepping_out(self, x0: float, spec: str, other_param: float, y: float, w: float, m: int) -> tuple:
        """
        Stepping-out procedure for generating an interval about a point x0 for slice sampling
        P
        -
        :parama x0: initial value
        :param spec: the variable being considered
        :param other_param: value of other variable under within density
        :param y: vertical level defining the slice
        :param w: estimate of the typical size of a slice
        :param m: integer limiting the size of the slice to m*w
        """
        U, V = uniform().rvs(2)
        L = x0 - w*U; R = L + w
        J = np.floor(m*V); K = m - 1 - J
        # Check for determing which parameter is being updated
        if spec != "loc": density = partial(self.pdf(loc=other_param))
        else: density = partial(self.pdf(scale=other_param))
        # 
        while J > 0 and y < density(other_param):
            L -= w; J -= 1
        while K > 0 and y < density(other_param):
            R += w; K -= 1
        return (L, R)
    def _shrinkage(self, x0: float, spec: str, other_param: float, y: float, interval: tuple) -> float:
        """
        Shrinkage procedure for generating a sample from the interlva produced in self._stepping_out()
        P
        -
        :parama x0: initial value
        :param spec: the variable being considered
        :param other_param: value of other variable under within density
        :param y: vertical level defining the slice
        :param interval: tuple of (L, R) defining interval to sample from
        """
        # Check for determing which parameter is being updated
        if spec != "loc": density = partial(self.pdf(loc=other_param))
        else: density = partial(self.pdf(scale=other_param))
        Lhat, Rhat = interval
        while True:
            U = uniform().rvs()
            x1 = Lhat + U * (Rhat - Lhat)
            if y < density(x1): 
                return x1
            if x1 < x0:
                Lhat = x1
            else:
                Rhat = x1

    def rvs(self, N: int=1) -> np.array:
        """
        Slice-sampling method for generating random variates
        https://agustinus.kristia.de/techblog/2015/10/24/slice-sampling/
        """
        return


# def lambda_nu_posterior(loc: float, scale: float, y: int, a: float, b: float, w: float) -> float:
#     """
#     Joint posterior distribution for (lambda, nu)
#     """
#     if loc > 0 and loc < w:
#         numerator = (y+a) * (b+np.log(factorial(y))) * loc**(y+a-1) * np.exp(-scale*(b+np.log(factorial(y))))
#         return numerator / (w**(y+a))
#     else:
#         return 0
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

if __name__ == "__main__":
    d = CMP_posterior(100, 0.1, 0.1, 0.1)
    X1 = np.linspace(0, 1, 10)
    X2 = np.linspace(0, 1, 10)
    Y = d.pdf(loc=X1, scale=X2)
    print(Y)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(X1, Y)
    plt.show()


