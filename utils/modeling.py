"""
Modeling distributions

"""
import numpy as np
from numpy.linalg import det, inv
from regression import count_party_votes, covariate_matrix
from scipy.stats import multivariate_normal, dirichlet

def multinomial_logit(X: np.array, A: np.array, B: np.array) -> np.array:
    """
    Multinomial logit model
    Assume that log-odds of each response variable follow a linear model
    P
    -
    X is nxp
    A is nx1
    B is nx1
    R
    -
    An nx1 response vector representing the log-odds of each response
    """
    return A + np.matmul(X.T, B)

# Posterior distributions


if __name__ == "__main__":
    X = covariate_matrix
    print(X)