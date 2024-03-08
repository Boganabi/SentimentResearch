
# this will find the differences between each model and the actual rating

import pandas as pd

df = pd.read_csv("model_evaluations.csv")

realValues = df["Real_Rating"]
totalVals = len(realValues)

rnnSuccesses = len(df[df["RNN_Class"] == realValues])

cnnSuccesses = len(df[df["CNN_Class"] == realValues])

lrSuccesses = len(df[df["Logistic_Regression_Class"] == realValues])

nbSuccesses = len(df[df["Naive_Bayes_Class"] == realValues])

rfSuccesses = len(df[df["Random_Forest_Class"] == realValues])

nltkSuccesses = len(df[df["NLTK_Class"] == realValues])

tbSuccesses = len(df[df["TextBlob_Class"] == realValues])

print("Number of successful predictions vs percentage of successful percentages:\n\
    RNN:", rnnSuccesses, 100 * rnnSuccesses / totalVals, "%\n\
    CNN:", cnnSuccesses, 100 * cnnSuccesses / totalVals, "%\n\
    Logistic Regression:", lrSuccesses, 100 * lrSuccesses / totalVals, "%\n\
    Naive Bayes:", nbSuccesses, 100 * nbSuccesses / totalVals, "%\n\
    Random Forest:", rfSuccesses, 100 * rfSuccesses / totalVals, "%\n\
    NLTK:", nltkSuccesses, 100 * nltkSuccesses / totalVals, "%\n\
    TextBlob:", tbSuccesses, 100 * tbSuccesses / totalVals, "%\n")

print("Total values:", totalVals)