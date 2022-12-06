"""
File for extracting and manipulating relevant data
"""
import pandas as pd
from data_constants import STATES, STATES_ABBR

# Global filepath
filepath = "./dataverse_files/1976-2020-senate.csv"

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data into a pandas dataframe
    """
    df = pd.read_csv(
        filepath, header=0, encoding="latin1", # no idea why the encoding is necessary
        usecols=["year", "state_po", "candidatevotes", "totalvotes", "party_detailed", "party_simplified"])
    return df

global_election_data = load_data(filepath)

def count_party_occurrences(party: str) -> float:
    """
    Get number of occurrences of a specific partyy
    """
    return sum([1 for p in df["party_detailed"] if p == party])

def count_party_votes(df: pd.DataFrame, year: int, state: str) -> tuple:
    """
    Get the number of votes a certain party recieved in a given election
    Returns a tuple of (party votes, total_votes)
    """
    global filepath
    df = load_data(filepath)
        
    rep_votes, dem_votes, lib_votes, other_votes = 0, 0, 0, 0
    vec=[0, 0, 0, 0]
    for _, row in df.iterrows():
        if row["year"] == int(year):
            if row["party_simplified"].upper() == "DEMOCRAT" and row['state_po']==state:
                dem_votes += row["candidatevotes"]
            if row["party_simplified"].upper() == "REPUBLICAN" and row['state_po']==state:
                rep_votes += row["candidatevotes"]
            if row["party_simplified"].upper() == "LIBERTARIAN" and row['state_po']==state:
                lib_votes += row["candidatevotes"]
            elif row['party_simplified']=='OTHER' and row['state_po']==state:
                other_votes += row["candidatevotes"]
    vec[0]=dem_votes
    vec[1]=rep_votes
    vec[2]=lib_votes
    vec[3]=other_votes
    
    #print(vec, state)
    return vec

def convert_irrelevant_parties(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
# def party_votes_df(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Create DataFrame containing votes received by each party, per state, per year
#     """ 
#     res = {"State": STATES_ABBR, "Republican": [], "Democrat": [], "Independent": [], "Other": []}
#     _, R, D, I, O = (r for r in res.keys())
#     for state in res["State"]:
#         for yr in [1976+i for i in range(2021-1976+1)]:
#             p_votes, t_votes = count_party_votes(df, "DEMOCRAT", str(state), yr)
#             res["Democrat"].append(p_votes)
#             p_votes, t_votes = count_party_votes(df, "REPUBLICAN", state, yr)
#             res["Republican"].append(p_votes)
#             p_votes, t_votes = count_party_votes(df, "LIBERTARIAN", state, yr)
#             res["Libertarian"].append(p_votes)
#             p_votes, t_votes = count_party_votes(df, "OTHER", state, yr)
#             res["Other"].append(p_votes)

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
    p, t = count_party_votes(df, "DEMOCRAT", "AZ", 1976)
    # party_votes_df(df)
    print(df)