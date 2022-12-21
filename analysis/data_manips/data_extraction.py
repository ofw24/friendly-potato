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

def load_slurm(data: str, filename: str) -> np.array:
    """
    Read in data from slurm output
    """
    print("Reading in data from SLURM output...\n")
    # Fun problem with floating point arithmetic!
    issue = "/home/chong21/sam_testing/metropolis/friendly-potato/analysis/mcmc/mcmc_methods.py:64: RuntimeWarning: divide by zero encountered in double_scalars\n  acceptance_ratio = (proposed_prob / prior_prob) / 10**5 # Scaling because of non-normalization"
    df, raw_data = load_data(data), pd.read_csv(data, sep="\t")
    countries = df["country"].unique()
    get_cont = lambda c: np.array(raw_data["continent"][raw_data["country"] == c])[0]
    res = dict([(c, [get_cont(c), z, p]) for c, z, p in zip(countries, [None for _ in countries], [0 for _ in countries])])
    # continents = ["Africa", "Americas", "Asia", "Europe", "Oceania"]
    with open(filename, "r") as slurm:
        lines = slurm.readlines()
        for ii, line in enumerate(lines):
            if any([c for c in countries if c in line]):
                newline = line.split(" is in ")
                small = newline[0] # country
            # if "SUPER CURSED OUTPUT THAT I MIGHT NEED" in line:
            #     # Getting predicted values
            #     preds = (str(lines[ii+13]).strip("\n") + str(lines[ii+14]).strip("\n")).replace("[", "]").replace("]", "")
            #     preds = [float(x) for x in preds.split() if x != ""]
            #     res[small][2] = preds
            if "Gibbs Beta" in line:
                gb = (str(lines[ii]).strip("\n") + str(lines[ii+1]).strip("\n")).replace("[", "]").replace("]", "")
                gb = np.array([g.split() for g in gb.split("Gibbs Beta : ")[1:]]).flatten()
                gb = list(map(float, gb))
                res[small][1] = gb
            if "Predicted gains" in line:
                pred = line.strip("Predicted gains : ").replace("[", "]").replace("]", "").strip(issue).split(", ")
                pred = list(map(float, pred))
                # print(pred)
                res[small][2] = pred

    return res

if __name__ == "__main__":
    data = "./raw_data/gapminder.tsv.txt"
    output = "./raw_data/slurm_10481503.out"
    # count = "Afghanistan"
    # le, leg = grab_life_exp_gain(df, count)
    # print(le)
    # print(leg)
    res = load_slurm(data, output)
    print(res["Germany"][1])