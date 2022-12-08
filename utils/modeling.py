"""
Modeling distributions

"""
import pickle
import numpy as np
from regression import get_X
from data_extraction import count_party_votes, global_election_data
from data_constants import CONST_YEARS, STATES, STATES_ABBR

def impute_data():
    """
    Impute missing points for voter data
    """
    pass

def pickle_me_timbers() -> None:
    X = [get_X(year) for year in CONST_YEARS]
    with open("observed_2.txt", "w") as outfile:
        for year in CONST_YEARS:
            v = ""
            for state in STATES_ABBR:
                vec = count_party_votes(global_election_data, year, state.upper()) # returns (dem, rep, lib, other)
                if sum(vec) != 0:
                    rat = vec[0] / (vec[0]+vec[1])
                    v += f"{vec[0]},{vec[1]},{vec[2]},{vec[3]}\n" # Write in the data as a string
                    print(f"{rat:0.8f} : {state} : {year}")
                else: # if missing data, impute
                    try:
                        v += "1,1,1,1\n"
                    except IndexError:
                        v += "1,1,1,1\n"
            outfile.write(v)
            outfile.write("NEW YEAR\n")
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

def hammer_and_pickle() -> None:
    Y = [] # place to store data for all election years
    # The jankiest way to load data you've ever seen
    election_index = 0
    with open("./observed_2.txt", "r") as infile:
        y = []
        for line in infile.readlines():
            try:
                data = [float(d) for d in line.strip("\n").split(",")]
                y.append(data)
            except ValueError:
                Y.append(y)
                election_index += 1 # Update counter if "NEW YEAR" is reached
                y = []
    return np.array(Y)

if __name__ == "__main__":
    import pandas as pd
    # pickle_me_timbers()
    Y = hammer_and_pickle() # len(Y) = 17 (years) // len(Y[0]) = 50 (states) // len(Y[0][0]) = 4 (parties) // len(Y[0][0][0]) = int
    observed = Y
    X = [get_X(year) for year in CONST_YEARS]
    beta_ols = ols_regressors(X, Y) # ERROR IS HERE
    # dg = pd.DataFrame(Y, columns=STATES)
    print(beta_ols)