
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# df = pd.read_csv("cleaned_data.csv")
df = pd.read_csv("../balanced_data.csv")

X_train = df["Text"]
y_train = df["Rating"]

text_clf = Pipeline([
    ('vect', CountVectorizer()), # convert text to vectors
    ('clf', LogisticRegression()), # naive bayes classifier
])

# train classifier
text_clf.fit(X_train, y_train)

# save model
joblib.dump(text_clf, 'sentiment_modelLR.pkl')