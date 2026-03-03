import mlflow.sklearn
import pandas as pd
import os

# Konfigurasi MLflow
mlflow.set_tracking_uri("http://localhost:5000")
MODEL_NAME = "Titanic_RF_Model"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

# Global variable untuk menyimpan model di memory (Lazy Loading)
_model = None

def load_model():
    """Meload model dari MLflow Registry ke memori."""
    global _model
    if _model is None:
        try:
            _model = mlflow.sklearn.load_model(MODEL_URI)
            print("Model berhasil di-load dari MLflow.")
        except Exception as e:
            print(f"Gagal meload model: {e}")
    return _model

def make_prediction(input_data: dict) -> int:
    """Menerima input dictionary, mengembalikan hasil prediksi."""
    model = load_model()
    if model is None:
        raise ValueError("Model belum tersedia di server MLflow.")
    
    # Konversi dictionary ke DataFrame 
    # (Scikit-Learn Pipeline membutuhkan DataFrame dengan nama kolom yang sesuai)
    df = pd.DataFrame([input_data])
    
    # Lakukan prediksi
    prediction = model.predict(df)[0]
    
    return int(prediction)