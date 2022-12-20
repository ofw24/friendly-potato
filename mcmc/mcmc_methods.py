"""
Script for extracting and manipulating data
"""
import numpy as np
import pandas as pd

def country_beta_ols(data: pd.DataFrame, country: str) -> np.array:
    """
    Get OLS estimate of linear regressors for a particular country
    """
    betas, life_exps = np.array([]), np.array([])
    # Vector of observations
    y = np.array(data[data["country"] == country]["lifeExp"])
     # Matrix containing data for only a particular country w/o the country name in the DataFrame
    X = data[data["country"] == country].drop(["country", "lifeExp"], axis=1)
    life_exps = np.append(life_exps, y)
    return np.linalg.lstsq(X, y, rcond=None)

def gibbs_sampler(data: pd.DataFrame, life_exp: np.array, beta_ols: np.array, N: int, burn: int) -> np.array:
    """
    Gibbs sampling algorithm
    """
    # Initialize prior variances
    var_life_exp, var_pop, var_gdp = np.var(life_exp), np.var(data["pop"]), np.var(data["gdpPercap"])


if __name__ == "__main__":
    from data_manips.data_extraction import load_data
    data = "../gapminder.tsv.txt"
    res = load_data(data)
    c, _, _, _ = country_beta_ols(res, "Afghanistan")
    print(c)