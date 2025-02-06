# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.model import train_model, inference
from ml.data import process_data 

# Add code to load in the data.
data = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Save the model, encoder, and label binarizer
with open("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("../model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
with open("../model/label_binarizer.pkl", "wb") as f:
    pickle.dump(lb, f)

# Example inference
preds = inference(model, X_test)
print("Sample predictions:", preds[:5])