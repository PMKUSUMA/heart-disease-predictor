import pytest
from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_page(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert b"CardioSense" in response.data


def test_health_api(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    json_data: dict = response.get_json()
    assert json_data["status"] == "ok"


def test_api_predict_missing_data(client) -> None:
    response = client.post("/api/predict", json={})
    assert response.status_code == 400


def test_api_predict_valid(client) -> None:
    payload: dict = {
        "age": 54, "sex": 1, "cp": 0, "trestbps": 122,
        "chol": 286, "fbs": 0, "restecg": 0, "thalach": 116,
        "exang": 1, "oldpeak": 3.2, "slope": 1, "ca": 2, "thal": 2
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    json_data: dict = response.get_json()
    assert "prediction" in json_data
    assert "confidence" in json_data