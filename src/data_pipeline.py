import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import config
import os

def run_data_pipeline():
    print("Menjalankan Data Pipeline (ETL)...")
    
    # 1. Load Raw Data
    df = pd.read_csv(config['data']['raw_path'])
    
    # 2. Filter Fitur & Hapus Baris Fatal (Target Kosong)
    features = config['model']['features']
    target = config['model']['target_col']
    df = df[features + [target]].dropna(subset=[target])
    
    # 3. Train-Test Split
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config['model']['random_state']
    )
    
    # 4. Simpan ke data/processed/
    os.makedirs(os.path.dirname(config['data']['processed_path']), exist_ok=True)
    
    # Gabungkan X_train dan y_train untuk disimpan sebagai reference.csv (untuk Evidently)
    reference_df = X_train.copy()
    reference_df[target] = y_train
    reference_df.to_csv(config['data']['processed_path'], index=False)
    
    print(f"Data Pipeline selesai. Data reference disimpan di: {config['data']['processed_path']}")
    
    # (Opsional) Anda bisa return X_train, X_test, y_train, y_test jika dipanggil langsung oleh train.py
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    run_data_pipeline()