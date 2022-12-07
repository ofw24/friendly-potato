"""
Modeling distributions

"""
import pickle
import numpy as np
from regression import get_X
from data_extraction import count_party_votes, global_election_data
from data_constants import STATES, STATES_ABBR

def impute_data():
    """
    Impute missing points for voter data
    """
    pass

def pickle_me_timbers():
    # Posterior distributions
    years = [i for i in range(1981, 2019)]
    years = [i for i in years if not i in [1993, 1994, 1995, 1996]]
    X = [get_X(year) for year in years]
    Y = []
    for year in years:
        v = []
        for state in STATES_ABBR:
            vec = count_party_votes(global_election_data, year, state.upper()) # returns (dem, rep, lib, other)
            if sum(vec) != 0:
                rat = vec[0] / (vec[0]+vec[1])
                v.append(rat)
                print(rat, state, year)
            else: # if missing data, impute
                try:
                    v.append(Y[-1])
                except IndexError:
                    v.append(1)
        Y.append(v)
    pickle.dump(Y, open("observed.pkl", "wb"))
    return np.array(X)

def ols_regressors(covariates: np.array, observations: np.array) -> np.array:
    """
    Get Ordinary-Least Squares (OLS) Estimate of the vector of regressors
    """
    # print(covariates)
    # print(observations)
    # Ordinary-Least Squares estimate of the regression vector
    XTX = np.array([np.matmul(x.T, x) for x in covariates])
    print(XTX.T[0].size) # XTX[0] is 9 x 102 
    XTy = [np.matmul(x.T, y) for x, y in zip(covariates, observations)]
    beta_ols = [np.matmul(np.linalg.inv(xtx), xty) for xtx, xty in zip(XTX, XTy)]
    return np.array(beta_ols)

if __name__ == "__main__":
    import pandas as pd
    X = pickle_me_timbers()
    # observed = pickle.load(open("observed.pkl", "rb")) # observed == Y
    Y = np.array(pickle.load(open("./observed.pkl", "rb")), dtype="object") # Y is 34 x 50
    # observed = Y
    # print(observed)
    years = [i for i in range(1981, 2019) if not i in [1993, 1994, 1995, 1996]]
    X = [get_X(year) for year in years]
    # beta_ols = ols_regressors(X, Y) # ERROR IS HERE
    dg = pd.DataFrame(Y, columns=STATES)
    print(dg)
    # print(beta_ols)