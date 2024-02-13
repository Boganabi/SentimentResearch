
# this file will balance the data so there arent so many 5 star reviews.

import pandas as pd
import numpy as np
import csv

fields = ["Rating", "Title", "Text"]
df = pd.read_csv("cleaned_data.csv")

removals = df.loc[df["Rating"] == 5.0]

# we have about 27k 5 star reviews, so lets remove 17k
drop_indicies = np.random.choice(removals.index, 17000, replace=False)
df = df.drop(drop_indicies)

# out to another csv file for safety :p
with open("balanced_data.csv", 'w', encoding="utf-8", newline="") as csvfile: # added newline="" due to funky windows stuff causing a new line to be written
    csvwriter = csv.writer(csvfile)

    # write title row
    csvwriter.writerow(fields)

    # write data
    # csvwriter.writerows(df)

df.to_csv("balanced_data.csv", sep=",", encoding="utf-8", index=False)