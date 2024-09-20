
import pandas as pd
from nltk import word_tokenize # to get number of unique words

# df = pd.read_csv("cleaned_data.csv")
df = pd.read_csv("balanced_data.csv")

# need to get number of different words for our model
data = df["Text"].map(word_tokenize).values
total_vocabulary = set(word.lower() for d in data for word in d) # create set of unique words
print("There are {} unique words in the dataset".format(len(total_vocabulary)))

# set target for our model
target = df["Rating"]

print(target.value_counts())

# find average length of review
res = sum(map(len, data))/float(len(data))
print(res)