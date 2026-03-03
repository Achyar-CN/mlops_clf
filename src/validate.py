import pandas as pd
import great_expectations as ge
from src.config import config

def validate_raw_data():
    print("Mulai validasi data mentah...")
    
    # Load data mentah ke dalam Great Expectations Pandas Dataset
    df_pandas = pd.read_csv(config['data']['raw_path'])
    df = ge.from_pandas(df_pandas)
    
    # 1. Ekspektasi: Kolom Pclass hanya boleh berisi 1, 2, atau 3
    exp_pclass = df.expect_column_values_to_be_in_set("Pclass", [1, 2, 3])
    
    # 2. Ekspektasi: Umur tidak boleh negatif (bisa null karena ditangani pipeline, tapi jika ada, harus > 0)
    exp_age = df.expect_column_values_to_be_between("Age", min_value=0.0, max_value=120.0, mostly=0.99)
    
    # 3. Ekspektasi: Sex hanya boleh 'male' atau 'female'
    exp_sex = df.expect_column_values_to_be_in_set("Sex", ["male", "female"])
    
    # Evaluasi hasil
    if not (exp_pclass["success"] and exp_age["success"] and exp_sex["success"]):
        print("Validasi Gagal! Periksa kembali data mentah Anda.")
        return False
    
    print("Validasi Sukses! Data siap digunakan untuk training/inferensi.")
    return True

if __name__ == "__main__":
    validate_raw_data()