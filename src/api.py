from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import csv
import os

# Import fungsi prediksi yang sudah dipisahkan
from src.predict import make_prediction
from src.config import config

app = FastAPI(title="Titanic Survival Prediction API", version="2.0")

# Skema input data API menggunakan Pydantic
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float

@app.post("/predict")
def predict_survival(passenger: Passenger):
    # 1. Lakukan Prediksi dengan memanggil fungsi dari predict.py
    try:
        input_dict = passenger.model_dump()
        prediction = make_prediction(input_dict)
    except ValueError as e:
        # Jika model belum dilatih atau MLflow mati
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat prediksi: {str(e)}")
    
    # 2. Logging untuk Continuous Monitoring (EvidentlyAI Drift Detection)
    # Tambahkan hasil prediksi ke dalam dictionary data yang masuk
    log_data = input_dict.copy()
    log_data['Survived'] = prediction
    
    current_data_path = config['data']['current_path']
    file_exists = os.path.isfile(current_data_path)
    
    # Pastikan folder tujuannya ada
    os.makedirs(os.path.dirname(current_data_path), exist_ok=True)
    
    # Tulis ke file current.csv
    with open(current_data_path, "a", newline='') as f:
        writer = csv.writer(f)
        # Ambil nama kolom (keys) dan nilainya (values)
        headers = list(log_data.keys())
        values = list(log_data.values())
        
        if not file_exists:
            writer.writerow(headers) # Tulis header jika file baru
        writer.writerow(values)      # Tulis baris datanya
        
    # 3. Kembalikan Response ke Pengguna
    return {"survived": prediction}