"""
Modeling distributions

"""
import numpy as np
from numpy.linalg import det, inv
from data_extraction import count_party_votes
from scipy.stats import multivariate_normal, dirichlet
from regression import get_X
from data_constants import STATES, STATES_ABBR
from data_extraction import global_election_data

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
years = [i for i in range(1982, 1983)]
years = [i for i in years if not i in [1993, 1994, 1995, 1996]]
X = [get_X(year) for year in years]
Y = []
for year in years:
    for state in STATES_ABBR:
        vec = count_party_votes(global_election_data, year, state.upper())
        # Y.append(vec[0]/(vec[0]+vec[1]))
        #print(vec[0])

# print(count_party_votes(global_election_data, 2000, 'MD'))

XTX = [np.matmul(x.T, x) for x in X]
XTy = [np.matmul(x.T, y) for x, y in zip(X, Y)]

# beta_ols = [np.matmul(np.inverse(xtx), xty) for xtx, xty in zip(XTX, XTy)]

# X=get_X(2000)

if __name__ == "__main__":
    ...
    #year=2000
    #X = get_X(2000)
    #print(X)