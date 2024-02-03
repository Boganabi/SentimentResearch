# this python file will take in a csv file of scraped reviews and clean it so it can be used for modeling

import csv
import string

import nltk # pip install --user -U nltk [used for removing stop words and filtering stem words]
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

FILENAME = "reviews.csv"
fields = ["Rating", "Title", "Text"]

cleanedRows = []

# read file
with open(FILENAME, 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:

        if len(row) > 0 and row[0] != 'Rating' and row[0] in ["1.0", "2.0", "3.0", "4.0", "5.0"]:
            # split by whitespace
            words = row[2].split() # defaults to space, use 2 as index to get actual review

            # remove punctuation and normalize case
            table = str.maketrans('', '', string.punctuation) # create table that removes any punctuation
            stripped = [w.translate(table).lower() for w in words]

            # use NLTK to remove stop words and filter stem words (get base form of word)
            stemmed = [porter.stem(word) for word in stripped if word not in stop_words]
            
            finalizedRow = [row[0]] + [row[1]] + [stemmed]
            
            cleanedRows.append(finalizedRow)

            # print(finalizedRow)

# print(cleanedRows)

with open("cleaned_data.csv", 'w', encoding="utf-8", newline="") as csvfile: # added newline="" due to funky windows stuff causing a new line to be written
    csvwriter = csv.writer(csvfile)

    # write title row
    csvwriter.writerow(fields)

    # write data
    csvwriter.writerows(cleanedRows)