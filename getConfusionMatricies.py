
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# load dataset
df = pd.read_csv("./model_evals_test2.csv")

ratings = df["Real_Rating"]
rnn = df["RNN_Class"]
cnn = df["CNN_Class"]
lr = df["Logistic_Regression_Class"]
nb = df["Naive_Bayes_Class"]
rf = df["Random_Forest_Class"]
nltk = df["NLTK_Class"]
tb = df["TextBlob_Class"]

# confusion matrices
# cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
cm = confusion_matrix(ratings, rnn)
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

cm = confusion_matrix(ratings, cnn)
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

cm = confusion_matrix(ratings, lr)
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

cm = confusion_matrix(ratings, nb)
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

cm = confusion_matrix(ratings, rf)
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

cm = confusion_matrix(ratings, nltk)
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

cm = confusion_matrix(ratings, tb)
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()