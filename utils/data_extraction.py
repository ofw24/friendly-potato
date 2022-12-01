"""
File for extracting and manipulating relevant data
"""
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data into a pandas dataframe
    """
    df = pd.read_csv(
        filepath, header=0, encoding="latin1", 
        usecols=["year", "state_po", "candidatevotes", "totalvotes", "party_detailed"])
    return df # no idea why the encoding is necessary

def count_party_occurrences(party: str) -> float:
    """
    Get number of occurrences of a specific partyy
    """
    return sum([1 for p in df["party_detailed"] if p == party])

def get_all_parties(df: pd.DataFrame) -> list:
    """
    Get all parties that people ran for Senate under from 1976-2020
    Assumes that DataFrame is loaded from '1976-2020-senate.csv'
    """
    all_entries = []
    for party in df["party_detailed"]:
        if party == party: all_entries.append(party) # get rid of NaNs
    entries_set = set(all_entries)
    if True:
        parties2 = []
        for party in entries_set:
            parties2.append((party, all_entries.count(party)))
    return all_entries, parties2

if __name__ == "__main__":
    filepath = "./dataverse_files/1976-2020-senate.csv"
    df = load_data(filepath)
    p, p2 = get_all_parties(df)
    for p in sorted(p2, key=lambda p: p[1], reverse=True):
        print(f"{p[0]:<40s} occurred {p[1]:<5.0f} times")