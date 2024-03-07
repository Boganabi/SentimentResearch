
# what i want to do is have the output csv have these columns:
# actual rating, rating text, RNN value, CNN value, LR value, NB value, RF value, NLTK value, TextBlob value

import pandas as pd
import csv
import joblib
import json
import string
import numpy as np

import nltk # pip install --user -U nltk [used for removing stop words and filtering stem words]
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob # pip install textblob
from keras.models import load_model
from keras.preprocessing import text, sequence

df = pd.read_csv("./balanced_data.csv")

ratings = df["Rating"]
texts = df["Text"]

numToClass = {
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive"
}

RNN_MODEL_FILE = "RNN/sentiment_modelRNN.keras"
RNN_TOKENIZER_FILE = "RNN/tokenizerRNN.json"
CNN_MODEL_FILE = "CNN/sentiment_modelCNN.keras"
CNN_TOKENIZER_FILE = "CNN/tokenizerCNN.json"
NB_MODEL_FILE = "NaiveBayes/sentiment_modelNB.pkl"
LR_MODEL_FILE = "LogisticRegression/sentiment_modelLR.pkl"
RF_MODEL_FILE = "RandomForest/sentiment_modelRF.pkl"

stop_words = set(stopwords.words('english'))

# load models
RNNMODEL = load_model(RNN_MODEL_FILE)
CNNMODEL = load_model(CNN_MODEL_FILE)
NBMODEL = joblib.load(NB_MODEL_FILE)
LRMODEL = joblib.load(LR_MODEL_FILE)
RFMODEL = joblib.load(RF_MODEL_FILE)

sia = SentimentIntensityAnalyzer()

def GetNNPreprocessedWords(sentence, tokenizerFilePath):
    # do preprocessing that was done with the training
    porter = PorterStemmer()
    # remove stop words and do stemming process, split by whitespace, remove punctuation, and normalize case
    words = sentence.split()
    table = str.maketrans('', '', string.punctuation) # create table that removes any punctuation
    stripped = [w.translate(table).lower() for w in words]
    stemmed = [porter.stem(word) for word in stripped if word not in stop_words]

    tokenizer = text.Tokenizer()
    with open(tokenizerFilePath) as f:
        data = json.load(f)
        tokenizer = text.tokenizer_from_json(data)

        tokenizer.word_index = { k.replace("'", ""): v for k, v in tokenizer.word_index.items() }

        tokenized_texts = tokenizer.texts_to_sequences([stemmed]) # convert text to integers that are usable by the model
        padded_input = sequence.pad_sequences(tokenized_texts, maxlen=400) # pads integers to be a certain length with 0's or by truncating
    return padded_input

def GetFloatClass(num):
    if num == 0: return "Neutral"
    if num < 0: return "Negative"
    if num > 0: return "Positive"

with open("model_evaluations.csv", 'w', encoding="utf-8", newline="") as csvfile: # added newline="" due to funky windows stuff causing a new line to be written
    csvwriter = csv.writer(csvfile)
    fields = ["Real_Rating", 
              "Text", 
              "RNN_raw", 
              "RNN_Class", 
              "CNN_raw", 
              "CNN_Class", 
              "Logistic_Regression_raw", 
              "Logistic_Regression_Class", 
              "Naive_Bayes_raw", 
              "Naive_Bayes_Class", 
              "Random_Forest_raw", 
              "Random_Forest_Class", 
              "NLTK_raw", 
              "NLTK_Class", 
              "TextBlob_raw",
              "TextBlob_Class"]

    # write title row
    csvwriter.writerow(fields)

    for index, t in texts.items():
        curr_row = []

        # add actual rating and text
        curr_row.append(numToClass[ratings[index]])
        curr_row.append(t)

        # get rating for RNN
        preprocessed = GetNNPreprocessedWords(t, RNN_TOKENIZER_FILE)
        prediction = RNNMODEL.predict(preprocessed, verbose=0)
        RNNRating = np.argmax(prediction) + 1 # since the result shifts it from 1->0 and 5 -> 4
        curr_row.append(RNNRating)
        curr_row.append(numToClass[RNNRating])

        # get rating for CNN
        preprocessed = GetNNPreprocessedWords(t, CNN_TOKENIZER_FILE)
        prediction = CNNMODEL.predict(preprocessed, verbose=0)
        CNNRating = np.argmax(prediction) + 1 # since the result shifts it from 1 -> 0 and 5 -> 4
        curr_row.append(CNNRating)
        curr_row.append(numToClass[CNNRating])

        # get rating for Logistic Regression
        LRRating = LRMODEL.predict([t])[0]
        curr_row.append(LRRating)
        curr_row.append(numToClass[LRRating])

        # get rating for Naive Bayes
        NBRating = NBMODEL.predict([t])[0]
        curr_row.append(NBRating)
        curr_row.append(numToClass[NBRating])

        # get rating for Random Forest
        RFRating = RFMODEL.predict([t])[0]
        curr_row.append(RFRating)
        curr_row.append(numToClass[RFRating])

        # get NLTK rating
        NLTKRating = sia.polarity_scores(t)["compound"]
        curr_row.append(NLTKRating)
        curr_row.append(GetFloatClass(NLTKRating))

        # get TextBlob rating
        BlobRating = TextBlob(t).sentiment.polarity
        curr_row.append(BlobRating)
        curr_row.append(GetFloatClass(BlobRating))

        csvwriter.writerow(curr_row)
        # break # temp for debugging