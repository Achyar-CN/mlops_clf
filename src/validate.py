import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from src.config import config

def validate_raw_data():
    print("Memulai validasi data menggunakan Pandera...")
    
    # 1. Baca data dengan Pandas biasa
    df = pd.read_csv(config['data']['raw_path'])
    
    # 2. Definisikan Skema Aturan (Rule)
    schema = DataFrameSchema({
        # Kolom Survived harus angka, isinya hanya boleh 0 atau 1
        "Survived": Column(int, Check.isin([0, 1])),
        
        # Kolom Age harus angka, nilainya antara 0-100, dan boleh kosong (nullable)
        "Age": Column(float, Check.between(0, 100), nullable=True),
        
        # Kolom Pclass harus angka 1, 2, atau 3
        "Pclass": Column(int, Check.isin([1, 2, 3]))
    })
    
    # 3. Lakukan Validasi
    try:
        validated_df = schema.validate(df)
        print("✅ Validasi berhasil! Data aman dan sesuai aturan.")
    except pa.errors.SchemaError as exc:
        print(f"❌ Validasi gagal pada kolom/aturan tertentu:\n{exc}")
        raise # Hentikan program jika data korup

if __name__ == "__main__":
    validate_raw_data()