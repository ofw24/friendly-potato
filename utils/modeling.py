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

def pickle_me_timbers() -> None:
    # Posterior distributions
    years = [i for i in range(1982, 2019)]
    years = [i for i in years[::2] if not i in [1993, 1994, 1995, 1996]]
    X = [get_X(year) for year in years]
    Y = []
    for year in years:
        v = []
        for state in STATES_ABBR:
            vec = count_party_votes(global_election_data, year, state.upper()) # returns (dem, rep, lib, other)
            if sum(vec) != 0:
                rat = vec[0] / (vec[0]+vec[1])
                v.append((vec[0], vec[1], vec[2], vec[3]))
                print(f"DDR : {rat:0.8f} : {state} : {year}")
            else: # if missing data, impute
                try:
                    v.append((2,2,2,2))
                except IndexError:
                    v.append((2,2,2,2))
        Y.append(v)
    with open("observed.txt", "w") as outfile:
        for y in Y:
            outfile.write(f"{y}\n")
    return

def ols_regressors(covariates: np.array, observations: np.array) -> np.array:
    """
    Get Ordinary-Least Squares (OLS) Estimate of the vector of regressors
    """
    # Ordinary-Least Squares estimate of the regression vector
    XTX = np.array([np.matmul(x.T, x) for x in covariates])
    print(XTX.T[0].size) # XTX[0] is 9 x 102 
    XTy = [np.matmul(x.T, y) for x, y in zip(covariates, observations)]
    beta_ols = [np.matmul(np.linalg.inv(xtx), xty) for xtx, xty in zip(XTX, XTy)]
    return np.array(beta_ols)

if __name__ == "__main__":
    import pandas as pd
    # pickle_me_timbers()
    # observed = pickle.load(open("observed.pkl", "rb")) # observed == Y
    Y = []
    # The jankiest way to load data you've ever seen
    with open("./observed.txt", "r") as infile:
        for index, line in enumerate(infile.readlines()):
            stupid_year = line.split("), "); year = []
            # print(line)
            for y in stupid_year:
                y += ")"; even_dumber_year = []
                y = y.lstrip("("); y = y.rstrip(")"); y = y.split(", ")
                # print(y)
                for a in y:
                    # print(a)
                    try:
                        even_dumber_year.append(float(a))
                    except ValueError:
                        if "[(" in a:
                            even_dumber_year.append(float(a.lstrip("[(")))
                        elif ")]\n" in a:
                            even_dumber_year.append(float(a.rstrip(")]\n")))
                year.append(even_dumber_year)
            Y.append(year)
    observed = Y
    years = [i for i in range(1982, 2019) if not i in [1993, 1994, 1995, 1996]]
    X = [get_X(year) for year in years[::2]]
    # print(Y)
    beta_ols = ols_regressors(X, Y) # ERROR IS HERE
    dg = pd.DataFrame(Y, columns=STATES)
    # print(dg)
    print(beta_ols)