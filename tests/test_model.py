import pytest
from unittest.mock import patch
from src.predict import make_prediction

# Kita palsukan fungsi load_model agar tidak mencari MLflow
@patch("src.predict.load_model")
def test_make_prediction_logic(mock_load_model):
    
    # 1. Buat Model Palsu (Dummy Model) untuk pengujian
    class DummyModel:
        def predict(self, df):
            # Asumsikan model ini selalu menebak 1 (selamat)
            return [1]
            
    # Ganti model asli dengan model palsu
    mock_load_model.return_value = DummyModel()
    
    # 2. Siapkan data uji
    input_data = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 25.0,
        "Fare": 75.5
    }
    
    # 3. Jalankan fungsi prediksi
    result = make_prediction(input_data)
    
    # 4. Verifikasi hasilnya
    assert isinstance(result, int), "Hasil prediksi harus berupa integer!"
    assert result in [0, 1], "Hasil prediksi harus 0 atau 1!"