# Read ./2023-2024/results.csv, sort according to finish, and print out a table that works with github markdown.

import pandas as pd
from pathlib import Path

# get the path to the results.csv file
results = Path("./2023-2024/results.csv")

# read the csv file into a dataframe
df = pd.read_csv(results)

# sort the dataframe by finish where 1 is the lowest number
df = df.sort_values(by=["Finish"])

# Only keep Display Name
df = df[["Display Name"]]

# Add a position column
df.insert(0, "Position", range(1, len(df) + 1))

# print the dataframe
print(df.to_markdown(index=False))
