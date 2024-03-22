
# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

import pandas as pd
import numpy as np
import nltk
import json
import io
import matplotlib.pyplot as plt
from nltk import word_tokenize # to get number of unique words
nltk.download("punkt")
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

# df = pd.read_csv("cleaned_data.csv")
df = pd.read_csv("../balanced_data.csv")

# need to get number of different words for our model
data = df["Text"].map(word_tokenize).values
total_vocabulary = set(word.lower() for d in data for word in d) # create set of unique words
print("There are {} unique words in the dataset".format(len(total_vocabulary)))

# set target for our model
target = df["Rating"]

print(target.value_counts())

# use one hot encoding bc we have a categorical target
y = pd.get_dummies(target).values

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(df["Text"]))
print("vocab size:", len(tokenizer.word_index))

tokenized_texts = tokenizer.texts_to_sequences(df["Text"]) # convert text to integers that are usable by the model
X = sequence.pad_sequences(tokenized_texts, maxlen=400) # pads integers to be a certain length with 0's or by truncating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train two traditional machine learning models such as Naive Bayes, Logistic Regression, and Random Forest
# and two more advanced deep learning models, such as CNN and RNN.
# create CNN model
model = Sequential()

#add model layers
embedding_size = 128 # can be tuned
model.add(Embedding(len(tokenizer.word_index) + 1, embedding_size, input_length=X.shape[1]))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='sigmoid'))

print(y.shape[1])
print(len(df["Rating"].unique()))
print(df["Rating"].unique())
print("shape of x train:", X_train.shape)
print("shape of x test:", X_test.shape)
print("shape of y train:", y_train.shape)
print("shape of y test:", y_test.shape)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary() # check the shape

model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# could test with confusion matrix, etc later if needed

y_pred = model.predict(X_test) # get our predictions

acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print("Overall accureacy of CNN: {:.3f}".format(acc))

cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

# save model
model.save("sentiment_modelCNN.keras")
tokenizer_json = tokenizer.to_json()
with io.open("tokenizerCNN.json", 'w', encoding="utf-8") as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))