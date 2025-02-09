import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
import os
from model import train_model, inference
from data import process_data


@pytest.fixture
def sample_data():
    data = pd.read_csv("../data/census.csv")
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    label = "salary"
    X_train, y_train, encoder, lb = process_data(
        data,
        categorical_features=categorical_features,
        label=label,
        training=True
    )
    model = train_model(X_train, y_train)
    return (
        data,
        categorical_features,
        label,
        X_train,
        y_train,
        encoder,
        lb,
        model)


def test_process_data(sample_data):
    """Test the process_data function."""
    data, categorical_features, label, _, _, _, _ = sample_data
    X, y, encoder, lb = process_data(
        data,
        categorical_features=categorical_features,
        label=label,
        training=True
    )
    assert X.shape[0] == data.shape[0]
    assert len(y) == data.shape[0]
    assert encoder is not None
    assert lb is not None


def test_train_model(sample_data):
    """Test the train_model function."""
    _, _, _, X_train, y_train, _, _, model = sample_data
    assert isinstance(model, RandomForestClassifier)
    assert len(model.classes_) > 1


def test_inference(sample_data):
    """Test the inference function."""
    _, _, _, X_train, y_train, _, _, model = sample_data
    preds = inference(model, X_train)
    assert len(preds) == len(y_train)
    assert preds[0] in model.classes_


if __name__ == "__main__":
    sys.path.append(os.path.abspath("ml"))
