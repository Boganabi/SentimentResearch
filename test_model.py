import string
import json
import numpy as np
import pandas as pd
import nltk # pip install --user -U nltk [used for removing stop words and filtering stem words]
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

df = pd.read_csv("cleaned_data.csv")

# load the model
loaded_model = load_model("sentiment_model.keras")

"""
1 star: I bought this product because of when we adopted a stray cat, she had fleas. I was confident this product would work and I immediately treated this new cat. I did not work but instead of realizing the issue was with the product, I assume it was my fault in the way that I had applied the product. So I reordered this product but this time I had to get enough for 3 cats since the 2 cats that we already had, had become infested with fleas due to the fact that the initial treatment on this new cat had been 100% INEFECTIVE. So after receiving the second order of this product we treated all 3 cats, while making an extreme effort to apply it completely and thoroughly. The result was 100% Ineffective for each one of the cats. Now we have a major flea problem directly caused by the fact that this product did not do what it is suppose to do. I am very disappointed to say the least with name ""Amazon basic"" which is the 'brand' of this product - which has had a serious negative impact on the daily life of our family which will takes weeks to overcome.
-> gave 0
2 star: Didn’t work at all. Cats have become immune to these ingredients. Went back to Advantage as it still works
-> gave 1
3 star: I don't think it's working. My feral cats I put it on are still scratching.
-> gave 2
4 star: I was wondering if anyone knows how long it actually takes before you can give them a bath and safely still be protected from fleas etc. I put this on my cat but around our 23 the next day I had to give her a bath. I’m wondering because it says 24 to 48 hours before it’s waterproof if I need to reapply or if we are still protected
-> gave 3
5 star: Fleas do not like this. Cat doesn't seem to mind at all. Product does what it is supposed to do. 
-> gave 4
"""

# user_input = "i dont understand bruh"
user_input = "I was wondering if anyone knows how long it actually takes before you can give them a bath and safely still be protected from fleas etc. I put this on my cat but around our 23 the next day I had to give her a bath. I’m wondering because it says 24 to 48 hours before it’s waterproof if I need to reapply or if we are still protected"

# preprocess input

# split by whitespace, remove punctuation, and normalize case
words = user_input.split()
table = str.maketrans('', '', string.punctuation) # create table that removes any punctuation
stripped = [w.translate(table).lower() for w in words]

# remove stop words and do stemming process
stemmed = [porter.stem(word) for word in stripped if word not in stop_words]

print(stemmed)

# do preprocessing that was done with the training
tokenizer = text.Tokenizer()
with open("tokenizer.json") as f:
    data = json.load(f)
    tokenizer = text.tokenizer_from_json(data)

    tokenizer.word_index = { k.replace("'", ""): v for k, v in tokenizer.word_index.items() }

    tokenized_texts = tokenizer.texts_to_sequences([stemmed]) # convert text to integers that are usable by the model
    padded_input = sequence.pad_sequences(tokenized_texts, maxlen=400) # pads integers to be a certain length with 0's or by truncating
    # print("Padded input:", padded_input)

    # make prediction
    prediction = loaded_model.predict(padded_input)
    print("predicted sentiment probabilities:", prediction)

    # get predicted sentiment class
    predicted_class = np.argmax(prediction)

    print("Predicted sentiment class:", predicted_class)