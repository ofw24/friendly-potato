"""
File for extracting and manipulating relevant data
"""
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data into a pandas dataframe
    """
    return pd.read_csv(filepath, header=0, encoding="latin1") # no idea why the encoding is necessary

if __name__ == "__main__":
    filepath = "../dataverse_files/1976-2020-senate.csv"
    df = load_data(filepath)
    print(df)