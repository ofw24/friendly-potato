"""
Script for extracting and manipulating data
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
from analysis.data_manips.data_extraction import ale

def country_beta_ols(data: pd.DataFrame, country: str) -> np.array:
    """
    Get OLS estimate of linear regressors for a particular country for all years
    """
    # Vector of observations
    y = np.array(data[data["country"] == country]["lifeExp"], dtype=int)
     # Matrix containing data for only a particular country w/o the country name in the DataFrame
    X = data[data["country"] == country].drop(["country", "lifeExp"], axis=1)
    return np.linalg.lstsq(X, y, rcond=None)

def ols_predicted_life_exps(data: pd.DataFrame, beta_ols: np.array, country: str) -> np.array:
    """ 
    Get predicted life expectencies for a given country for all years
    """
    X = data[data["country"] == country].drop(["country", "lifeExp"], axis=1)
    return np.matmul(X, beta_ols)

def linear_fit_params_life_exps(df: pd.DataFrame, country: str) -> tuple:
    """
    Fit linear regression to life expectancy data for a particular country
    Returns a function f: R^1 -> R^1 and the correlation coefficient for the fit
    """
    year = ale(df, country, "year")
    life_exp = ale(df, country)
    res = linregress(year, life_exp)
    # lambda x: x*res.slope + res.intercept
    return res.slope, res.intercept, res.rvalue


def residuals(data: pd.DataFrame, country: str) -> tuple:
    """
    Get residuals of predicted life expectency versus actual for a particular country across all years
    """
    beta_ols, _, _, _ = country_beta_ols(data, country)
    ols_predicted = ols_predicted_life_exps(data, beta_ols, country)
    y = np.array(data[data["country"] == country]["lifeExp"])
    return np.absolute(y - ols_predicted)

def data_spline(actuals: np.array, predicted: np.array, order: int=3, knots: float=None) -> UnivariateSpline:
    """
    Fit spline to residuals
    Returns a function f: R^1 -> R^1
    """
    residuals = np.absolute(actuals - predicted) # Y
    return UnivariateSpline(actuals, residuals, k=order, s=knots)

def double_logistic(params: np.array, actual_life_exp: float) -> float:
    """
    Double logistic regression model
    """
    A1, A2 = 4.4, 0.5
    d1, d2, d3, d4, k, z = params
    first  = k / ( 1 + np.exp(-(A1/d2) * (actual_life_exp-d1-A2/d2)) )
    second = (z - k) / ( 1 + np.exp(-(A1/d4) * (actual_life_exp-A2/d4)) )
    return -(first + second)

def double_logistic_function(params: np.array) -> float:
    """
    Double logistic regression model function
    """
    A1, A2 = 4.4, 0.5
    d1, d2, d3, d4, k, z = params
    print(f"d1 : {d1}\nd2 : {d2}\nd3 : {d3}\nd4 : {d4}\nk  : {k}\nz  : {z}")
    # d1, d2, d4 = np.log(d1), np.log(d2), np.log(d4)
    # print(f"log(d1) : {d1}\nlog(d2) : {d2}\nd3      : {d3}\nlog(d4) : {d4}\nk       : {k}\nz       : {z}")
    su = sum([d1, d2, d3, d4])
    f = lambda life_exp: k / ( 1 + np.exp( -(2*np.log(9)/d1) * (life_exp-su+0.5*d1) ) )
    s = lambda life_exp: (z - k) / ( 1 + np.exp( -(2*np.log(9)/d3) * (life_exp-d4-0.5*d3) ) )
    return lambda a: -(f(a) + s(a))

if __name__ == "__main__":
    pass