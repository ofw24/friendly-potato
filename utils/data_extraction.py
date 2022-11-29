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

def get_all_parties(df: pd.DataFrame) -> list:
    """
    Get all parties that people ran for Senate under from 1976-2020
    """
    parties = []
    for party in df["party_detailed"]:
        if party == party: # get rid of NaNs
            # if party not in parties: parties.append(party)
            parties.append(party)
    if True:
        parties2 = []
        for party in set(parties):
            parties2.append((party, parties.count(party)))
    return parties, parties2

if __name__ == "__main__":
    filepath = "./dataverse_files/1976-2020-senate.csv"
    df = load_data(filepath)
    p, p2 = get_all_parties(df)
    for p in sorted(p2):
        print(f"Party {p[0]:<50s} has {p[1]:<5.0f} occurences")