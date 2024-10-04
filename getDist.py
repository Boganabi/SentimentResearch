
# import pandas as pd
# from nltk import word_tokenize # to get number of unique words

# df = pd.read_csv("cleaned_data.csv")

# # need to get number of different words for our model
# data = df["Text"].map(word_tokenize).values
# total_vocabulary = set(word.lower() for d in data for word in d) # create set of unique words
# print("There are {} unique words in the cleaned dataset".format(len(total_vocabulary)))

# # set target for our model
# target = df["Rating"]

# print("Distribution for each rating in cleaned dataset: ")
# print(target.value_counts())

# res = sum(map(len, data))/float(len(data))

# print("Average length of text: " + str(res))

# df = pd.read_csv("balanced_data.csv")

# data = df["Text"].map(word_tokenize).values
# total_vocabulary = set(word.lower() for d in data for word in d) # create set of unique words
# print("There are {} unique words in the balanced dataset".format(len(total_vocabulary)))

# target = df["Rating"]

# print("Distribution for each rating in balanced dataset: ")
# print(target.value_counts())

# res = sum(map(len, data))/float(len(data))

# print("Average length of text: " + str(res))

import csv

FILENAME = "reviews.csv"

currsum = 0
count = 0

# read file
with open(FILENAME, 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:

        if len(row) > 0 and row[0] in ["1.0", "2.0", "3.0", "4.0", "5.0"]:
            # find length of row
            currsum += len(row[2])
            count += 1

print("Average length:")
print(currsum / count)