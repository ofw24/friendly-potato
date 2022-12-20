"""
Script for extracting and manipulating data
"""
import pandas as pd

def cont_2_color(continent: str) -> str:
    transform = {"Africa": "r", "Americas": "b", "Asia": "g", "Europe": "m", "Oceania": "c"}
    return transform[continent]

def load_data(filename: str) -> pd.DataFrame:
    """
    Load data into a pd.DataFrame
    """
    rawdata = pd.read_csv(filename, sep="\t")
    continent = pd.get_dummies(rawdata["continent"], drop_first=True, prefix="continent")
    data = rawdata[["year", "country", "pop", "gdpPercap", "lifeExp"]]
    return pd.concat([data, continent], axis=1)

def grab_series(filename: str, series: str) -> pd.Series:
    """
    Get particular data series from raw data
    """
    rawdata = pd.read_csv(filename, sep="\t")
    return rawdata[series]

if __name__ == "__main__":
    data = "gapminder.tsv.txt"
    res = load_data(data)
    print(res)