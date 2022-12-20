"""
Script for extracting and manipulating data
"""
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

def ale(df: pd.DataFrame, country: str) -> np.array:
    """
    Get actual life expectencies for a particular country
    """
    return np.array(df[df["country"] == country]["lifeExp"])

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
    A1, A2 = 17.6, 0.125
    d1, d2, d3, d4, k, z = params
    first  = k / ( 1 + np.exp(-(A1*d2) * (actual_life_exp-d1-A2*d2)) )
    second = (z - k) / ( 1 + np.exp(-(A1*d4/d3) * (actual_life_exp-A2*d4)) )
    return -(first + second)

if __name__ == "__main__":
    pass