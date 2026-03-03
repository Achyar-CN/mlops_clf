from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api import app

client = TestClient(app)

# @patch akan memalsukan (mock) fungsi make_prediction agar TIDAK menghubungi MLflow
@patch("src.api.make_prediction")
def test_predict_endpoint_valid_input(mock_predict):
    # Kita paksa fungsi palsu ini untuk selalu mengembalikan nilai 1 (Selamat)
    mock_predict.return_value = 1
    
    payload = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 25.0,
        "Fare": 75.5
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] == 1

def test_predict_endpoint_invalid_input():
    # Simulasi input data yang salah (tipe data keliru, misal Age diisi string)
    payload = {
        "Pclass": 1,
        "Sex": "female",
        "Age": "Dua Puluh Lima",  # Sengaja dibuat salah untuk memicu error Pydantic
        "Fare": 75.5
    }
    response = client.post("/predict", json=payload)
    
    # Harus ditolak oleh FastAPI/Pydantic dengan error 422 Unprocessable Entity
    assert response.status_code == 422