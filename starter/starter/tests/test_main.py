from fastapi.testclient import TestClient
from main import app

# Create a test client
client = TestClient(app)


def test_get_root():
    """Test the GET / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the ML Model Inference API"
    }, "GET / response did not match expected message"


def test_post_predict_high_income():
    """Test the POST /predict endpoint when predicting >50k."""
    input_data = {
        "workclass": "Private",
        "education": "Doctorate",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
        "age": 50,
        "fnlwgt": 77516,
        "education-num": 13,
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == ">50K", (
        f"Expected '>50K' but got {response.json()['prediction']}"
    )


def test_post_predict_low_income():
    """Test the POST /predict endpoing when predicting <=50k"""
    input_data = {
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
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == "<=50K", (
        f"Expected '<=50K' got {response.json()['prediction']}"
    )
