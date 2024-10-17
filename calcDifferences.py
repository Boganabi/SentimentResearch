# this will find the differences between each model and the actual rating

import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv("model_evals_test2.csv")

realValues = df["Real_Rating"]

print("RNN classification report: ")
print(classification_report(realValues, df["RNN_Class"]))

print("CNN classification report: ")
print(classification_report(realValues, df["CNN_Class"]))

print("LR classification report: ")
print(classification_report(realValues, df["Logistic_Regression_Class"]))

print("NB classification report: ")
print(classification_report(realValues, df["Naive_Bayes_Class"]))

print("RF classification report: ")
print(classification_report(realValues, df["Random_Forest_Class"]))

print("NLTK classification report: ")
print(classification_report(realValues, df["NLTK_Class"]))

print("TextBlob classification report: ")
print(classification_report(realValues, df["TextBlob_Class"]))

# totalVals = len(realValues)

# numPos = len(df[df["Real_Rating"] == "Positive"])
# numNeu = len(df[df["Real_Rating"] == "Neutral"])
# numNeg = len(df[df["Real_Rating"] == "Negative"])

# rnnSuccesses = len(df[df["RNN_Class"] == realValues])

# cnnSuccesses = len(df[df["CNN_Class"] == realValues])

# lrSuccesses = len(df[df["Logistic_Regression_Class"] == realValues])

# nbSuccesses = len(df[df["Naive_Bayes_Class"] == realValues])

# rfSuccesses = len(df[df["Random_Forest_Class"] == realValues])

# nltkSuccesses = len(df[df["NLTK_Class"] == realValues])

# tbSuccesses = len(df[df["TextBlob_Class"] == realValues])

# print("Number of successful predictions vs percentage of successful percentages:\n\
#     RNN:", rnnSuccesses, 100 * rnnSuccesses / totalVals, "%\n\
#     CNN:", cnnSuccesses, 100 * cnnSuccesses / totalVals, "%\n\
#     Logistic Regression:", lrSuccesses, 100 * lrSuccesses / totalVals, "%\n\
#     Naive Bayes:", nbSuccesses, 100 * nbSuccesses / totalVals, "%\n\
#     Random Forest:", rfSuccesses, 100 * rfSuccesses / totalVals, "%\n\
#     NLTK:", nltkSuccesses, 100 * nltkSuccesses / totalVals, "%\n\
#     TextBlob:", tbSuccesses, 100 * tbSuccesses / totalVals, "%\n")

# print("Total values:", totalVals)
# print("Positive points:", numPos)
# print("Neutral points:", numNeu)
# print("Negative points:", numNeg)