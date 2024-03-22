
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

# df = pd.read_csv("cleaned_data.csv")
df = pd.read_csv("../balanced_data.csv")

X = df["Text"]
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

text_clf = Pipeline([
    ('vect', CountVectorizer()), # convert text to vectors
    ('clf', RandomForestClassifier()), # naive bayes classifier
])

# train classifier
text_clf.fit(X_train, y_train)

acc = text_clf.score(X_test, y_test)
print("Model score for Random Forest:", acc)

y_pred = text_clf.predict(X_test)

# cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

# save model
joblib.dump(text_clf, 'sentiment_modelRF.pkl')