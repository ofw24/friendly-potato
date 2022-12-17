import os
import pandas as pd


# data = pd.DataFrame()
# for filename in os.listdir("dataverse_files/income"):
#    with open(os.path.join("dataverse_files/income", filename), 'r') as f: # open in readonly mode
#         data = pd.concat([data, pd.read_csv(f)['Value']], axis=1)
#         print(f)
# print(data)
# data.to_csv("dataverse_files/unemployment/all_unemployment.csv")

data=pd.read_csv("dataverse_files/income/Book1.csv")
print(data.T)