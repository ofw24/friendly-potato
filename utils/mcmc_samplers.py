"""
Script for storing various Markov Chain Monte Carlo sampling methods
Many of these are case-specific and need to be modified in the future for more general use

Authors
Sam Dawley & Oliver Wolff
"""
import numpy as np
from abc import ABC

class mcmc_sampler(ABC):
    """
    Base class for Markov Chain Monte Carlo sampling methods
    """
    pass

class gibbs_sampler(mcmc_sampler):
    """
    Gibbs sampling algorithm for getting posterior distributions
    """
    def __init__(self, 
        starting_points: np.array, 
        conditional_distributions: np.array, 
        parameter_update_functions: np.array,
        N: int, burn: int
        ):
        """
        Note that 'conditional_distributions', 'parameter_update_functions', etc. should be ordered in the same manner,
        i.e., if conditional_distributions = [theta, gamma, sigma, ...] then parameter_update_functions = [theta, gamma, sigma, ...]
        """
        self.priors = starting_points
        self.full_cond_posts = conditional_distributions
        self.new_param_funcs = parameter_update_functions
        self.N = N
        self.burn = burn
    def _update_parameters(self):
        """
        Get new parameters of full conditional posterior distributions after each iteration
        """
        new_params = np.zeros(len(self.full_cond_posts))
        for func in self.new_param_funcs:
            new_params += func()

    def sample(self, observed_data: np.array, prior_parameters: np.array):
        """
        Run iterations of the Gibbs sampler
        """
        # Create arrays for storing posterior distributions of parameters
        posterior_arrays = np.array([])
        for _ in self.full_cond_posts:
            posterior_arrays = np.append(posterior_arrays, np.zeros(self.N-self.burn))
        # Iterate
        for t in range(self.N):
            print(f"GIBBS SAMPLER ITERATION {t+1}")
            # Update parameters
            new_params = self._update_parameters()






# Gibbs sampling algorithm
def gibbs_sampler(
    t1_conditional: norm,
    t2_conditional: norm,
    N: int, burn: int
    ) -> tuple:
    """
    ...

    Parameters
    ----------
    t1_conditional (callable) full conditional posterior distribution of theta_1
    t2_conditional (callable) full conditional posterior distribution of theta_2
    N (int) is the number of iterations for the sampler
    burn (int) is the burn-in period

    Returns 
    -------
    t1_posterior (np.array) is the empirical posterior distribution of theta_1
    t2_posterior (np.array) is the empirical posterior distribution of theta_2
    """
    # Global variables from given data
    global a1, a2, b1, b2, sample_data, n
    # Create arrays for storing posterior distributions of parameters
    t1_posterior, t2_posterior = np.zeros(N-burn), np.zeros(N-burn)
    # Create prior parameter for initialization
    t1_prior, t2_prior = 25, 25
    # Generate some parameters which are unchanging with each iteration
    ensemble_sum = sum(sample_data)
    t1_scale, t2_scale = 1 / (1/b1**2 + n), 1 / (1/b2**2 + n)
    t1_partial_loc, t2_partial_loc = a1/b1**2 + ensemble_sum, a2/b2**2 + ensemble_sum
    # Update parameters from full conditional distributions and append to posterior arrays
    for t in range(N):
        print(f"GIBBS SAMPLER ITERATION {t+1}")
        # Generate posterior parameters and update parameters
        # For theta_1
        t1_loc = (t1_partial_loc - n*t2_prior) / t1_scale
        t1_post = t1_conditional(loc=t1_loc, scale=t1_scale).rvs()
        test = np.random.rand()
        # test = 1
        if test > 0.7: # Implement random updates (quasi-metropolis-hastings, if you will)
            t1_prior = t1_post
        # For theta_2
        t2_loc = (t2_partial_loc - n*t1_prior) / t2_scale
        t2_post = t2_conditional(loc=t2_loc, scale=t2_scale).rvs()
        if test > 0.7:
            t2_prior = t2_post
        # If we're past the burn-in period, append realizations of the parameters to posterior arrays
        if t >= burn:
            t1_posterior[t-burn] += t1_post
            t2_posterior[t-burn] += t2_post
    return t1_posterior, t2_posterior





if __name__ == "__main__":
    ...