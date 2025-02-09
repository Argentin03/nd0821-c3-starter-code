import pytest
from fastapi.testclient import TestClient
from main import app  # Import the FastAPI app

# Create a test client
client = TestClient(app)


def test_get_root():
    """Test the GET / endpoint."""
    response = client.get("/")
    assert response.status_code == 200


@pytest.mark.parametrize("input_data, expected_output", [
    # Test Case 1: Expecting ">50K"
    ({
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
    }, ">50K"),
    
    # Test Case 2: Expecting "<=50K"
    ({
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
    }, "<=50K"),
])
def test_post_predict(input_data, expected_output):
    """Test the POST /predict endpoint with different input cases."""
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == expected_output
