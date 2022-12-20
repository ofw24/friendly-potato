"""
Script for extracting and manipulating data
"""
import numpy as np
import pandas as pd
from itertools import pairwise

def ale(df: pd.DataFrame, country: str, param: str="lifeExp") -> np.array:
    """
    Get actual life expectencies for a particular country
    """
    return np.array(df[df["country"] == country][param])

def get_continent(df: pd.DataFrame, country: str) -> np.array:
    """
    Get actual values of a particular parameter for a particular country
    """
    return np.array(df["continent"][df["country"] == country])[0]

def get_all_continents(df: pd.DataFrame, continent: str) -> np.array:
    """
    Get actual values of a particular parameter for a particular country
    """
    return np.array(df["country"][df["continent"] == continent])

def grab_life_exp_gain(df: pd.DataFrame, country: str) -> tuple:
    """
    Get actual gain in life expectancy between each year for a particular country
    """
    life_exps = ale(df, country)
    return life_exps[1:], np.array([p[1] - p[0] for p in pairwise(life_exps)])

def grab_series(filename: str, series: str) -> pd.Series:
    """
    Get particular data series from raw data
    """
    rawdata = pd.read_csv(filename, sep="\t")
    return rawdata[series]

def load_data(filename: str) -> pd.DataFrame:
    """
    Load data into a pd.DataFrame
    """
    rawdata = pd.read_csv(filename, sep="\t")
    continent = pd.get_dummies(rawdata["continent"], drop_first=True, prefix="continent")
    data = rawdata[["year", "country", "pop", "gdpPercap", "lifeExp"]]
    return pd.concat([data, continent], axis=1)

if __name__ == "__main__":
    data = "gapminder.tsv.txt"
    df = load_data(data)
    count = "Afghanistan"
    le, leg = grab_life_exp_gain(df, count)
    print(le)
    print(leg)