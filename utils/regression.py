"""
Organizing data for regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_constants import STATES
import sys
sys.path.append('../utils')

def vote_skew(election_df: pd.DataFrame, year: int, state: str) -> float:
    """
    Finds the skew of election results within a given state
    1  -> all votes for democrats within a state
    -1 -> all votes for republicans within a state
    """
    year_range = [year-i for i in range(1, 7)]
    election = election_df[election_df['year'].isin(year_range)]
    election = election[election['state']==state.upper()]
    skew = (election[election['party_simplified']=='DEMOCRAT']['candidatevotes'].tolist()[0]-election[election['party_simplified']=='REPUBLICAN']['candidatevotes'].tolist()[0])/election[election['party_simplified']=='REPUBLICAN']['totalvotes'].tolist()[0]
    return skew

def unemployment_by_state(unemployment: pd.DataFrame, states: list, state: str, year: int) -> float:
    """
    Gets the unemployment info by state
    """
    unemployment = unemployment.iloc[(year-1976)*12:(year-1975)*12]
    index = states.index(state)+1
    unemployment = unemployment.iloc[:, index]
    return np.mean(unemployment)

def get_X(year: int) -> np.array:
    print('getting X for ', year)
    # read in data
    election = pd.read_csv("./dataverse_files/1976-2020-senate.csv", encoding="latin1")
    income_historical = pd.read_csv("./dataverse_files/income/59 to 89 household.csv")
    income_2010 = pd.read_csv("./dataverse_files/income/2010.csv")
    income_2019 = pd.read_csv("./dataverse_files/income/2019.csv")
    unemployment_data = pd.read_csv("./dataverse_files/unemployment/all_unemployment.csv")

    # get relevant info for the year in question
    prev_election = election[election['year']==year-6]
    for i in range(year-6, year):
        prev_election = pd.concat([prev_election, election[election['year']==i]])

    prev_skew = [vote_skew(prev_election, year, state) for state in STATES]

    if 1976 <= year and year < 1979:
        prev_income = income_historical["Current dollars 1969"]
    elif 1979 <= year and year < 1989:
        prev_income = income_historical["Current dollars 1979"]
    elif 1989 <= year and year < 2010:
        prev_income = income_historical["Current dollars 1989"]
    elif 2010 <= year and year < 2019:
        prev_income = income_2010
    else:
        prev_income = income_2019

    # print(int(income_2010['Alabama'].tolist()[0].replace(',', '')))

    if year<2010:
        X = [[float(vote_skew(prev_election, year, state)),
                float(unemployment_by_state(unemployment_data, STATES, state, year)),
                float((prev_income[STATES.index(state)+1]).replace(",", ""))] for state in STATES]
        covariate_matrix = np.array(X)
    else:
        X = [[float(vote_skew(prev_election, year, state)),
                float(unemployment_by_state(unemployment_data, STATES, state, year)),
                float(prev_income[state].tolist()[0].replace(',', ''))] for state in STATES]
        covariate_matrix = np.array(X)       

    return covariate_matrix

#get_X(2010)
# if __name__ == "__main__":
#     print(get_X(2001))
