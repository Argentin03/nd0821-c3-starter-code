from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import uvicorn
import sys
import os
from data import process_data
from model import inference

# Initialize FastAPI app
app = FastAPI()

# Load trained model, encoder, and label binarizer
with open("./model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("./model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("./model/label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

# Define categorical features
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


# Define a Pydantic model to handle incoming requests
class InputData(BaseModel):
    workclass: str = Field(..., example="Private")
    education: str = Field(..., example="Bachelors")
    marital_status: str = Field(...,
                                alias="marital-status",
                                example="Never-married")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    native_country: str = Field(...,
                                alias="native-country",
                                example="United-States")
    age: int = Field(..., example=39)
    fnlwgt: int = Field(..., example=77516)
    education_num: int = Field(..., alias="education-num", example=13)
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "examples": [
                {
                    "workclass": "Private",
                    "education": "Bachelors",
                    "marital-status": "Never-married",
                    "occupation": "Prof-specialty",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "native-country": "United-States",
                    "age": 39,
                    "fnlwgt": 77516,
                    "education-num": 13,
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40
                },
                {
                    "workclass": "State-gov",
                    "education": "Some-college",
                    "marital-status": "Married-civ-spouse",
                    "occupation": "Adm-clerical",
                    "relationship": "Husband",
                    "race": "Black",
                    "sex": "Male",
                    "native-country": "United-States",
                    "age": 28,
                    "fnlwgt": 336951,
                    "education-num": 10,
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 30
                }
            ]
        }


@app.get("/")
def root():
    """Root GET endpoint that returns a welcome message."""
    return {"message": "Welcome to the ML Model Inference API"}


@app.post("/predict")
def predict(input_data: InputData) -> dict:
    """POST endpoint for model inference."""

    # Convert input data to DataFrame
    data_dict = input_data.dict(by_alias=True)
    data_df = pd.DataFrame([data_dict])

    # Process input data
    X_processed, _, _, _ = process_data(
        data_df, categorical_features=cat_features,
        label=None, training=False, encoder=encoder, lb=lb
    )

    # Run model inference
    prediction = inference(model, X_processed)

    # Convert numeric output to human-readable labels
    predicted_label = lb.inverse_transform(prediction)[0]

    return {"prediction": predicted_label}


if __name__ == "__main__":
    sys.path.append(os.path.abspath("starter/ml"))
    uvicorn.run(app, host="0.0.0.0", port=8000)
