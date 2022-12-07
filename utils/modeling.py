"""
Modeling distributions

"""
import pickle
import numpy as np
from regression import get_X
from data_extraction import count_party_votes, global_election_data
from data_constants import STATES, STATES_ABBR

def impute_votes(df) -> None:
    """
    Impute missing points from dataframe containing voting data
    """
    ...
    return

def data_dump():
    # Posterior distributions
    years = [i for i in range(1981, 2019)]
    years = [i for i in years if not i in [1993, 1994, 1995, 1996]]
    X = [get_X(year) for year in years]
    Y = []
    for year in years:
        v = []
        for state in STATES_ABBR:
            vec = count_party_votes(global_election_data, year, state.upper()) # (dem, rep, lib, other)
            if sum(vec) != 0:
                v.append(vec[0]/(vec[0]+vec[1]))
                print(vec[0]/(vec[0]+vec[1]), state, year)
            else:
                v.append(2)
        Y.append(v)
    pickle.dump(Y, open("observed.pkl", "wb"))

    # Ordinary-Least Squares estimate of the regression vector
    XTX = [np.matmul(x.T, x) for x in X]
    XTy = [np.matmul(x.T, y) for x, y in zip(X, Y)]
    beta_ols = [np.matmul(np.linalg.inv(xtx), xty) for xtx, xty in zip(XTX, XTy)]
    print(f"Function {beta_ols}")
    return beta_ols


if __name__ == "__main__":
    # beta_OLS = data_dump()
    import pandas as pd
    observed = pickle.load(open("observed.pkl", "rb")) # observed == Y
    years = [i for i in range(1981, 2019)]
    years = [i for i in years if not i in [1993, 1994, 1995, 1996]]
    X = [get_X(year) for year in years]
    XTX = [np.matmul(x.T, x) for x in X]
    XTy = [np.matmul(x.T, y) for x, y in zip(X, observed)]
    beta_ols = [np.matmul(np.linalg.inv(xtx), xty) for xtx, xty in zip(XTX, XTy)]
    print(f"Observed {beta_ols}")
    dg = pd.DataFrame(observed, columns=STATES)
    print(dg)