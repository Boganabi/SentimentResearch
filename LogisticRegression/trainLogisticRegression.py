
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# df = pd.read_csv("cleaned_data.csv")
df = pd.read_csv("../balanced_data.csv")

X = df["Text"]
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

text_clf = Pipeline([
    ('vect', CountVectorizer()), # convert text to vectors
    ('clf', LogisticRegression()), # naive bayes classifier
])

# train classifier
text_clf.fit(X_train, y_train)

acc = text_clf.score(X_test, y_test)
print("Model score for Logistic Regression:", acc)

# save model
joblib.dump(text_clf, 'sentiment_modelLR.pkl')