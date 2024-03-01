
import joblib

# MODEL_FILE = "NaiveBayes/sentiment_modelNB.pkl"
# MODEL_FILE = "LogisticRegression/sentiment_modelLR.pkl"
MODEL_FILE = "RandomForest/sentiment_modelRF.pkl"

# load saved model
loaded_model = joblib.load(MODEL_FILE)

# test input
test_data = ["works great now"]
predictions = loaded_model.predict(test_data)

print(predictions)