import yaml
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Import Data Pipeline yang sudah kita pisahkan
from src.data_pipeline import run_data_pipeline
from src.config import config

# Konfigurasi MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(config['mlflow']['experiment_name'])

def train_model():
    print("Memulai proses training...")
    
    # 1. Jalankan Data Pipeline untuk mendapatkan data yang siap latih
    X_train, X_test, y_train, y_test = run_data_pipeline()

    # 2. Buat Preprocessing Pipeline (Scikit-Learn)
    numeric_features = ["Age", "Fare"]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_features = ["Sex"]
    categorical_transformer = Pipeline(steps=[
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # 3. Gabungkan Preprocessor dan Model ke dalam satu Pipeline Utama
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=config['model']['n_estimators'], 
            random_state=config['model']['random_state']
        ))
    ])

    # 4. MLflow Tracking
    with mlflow.start_run():
        print("Melatih model dan mencatat ke MLflow...")
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        
        # Hitung metrics evaluasi
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        # Log parameter & metrics
        mlflow.log_param("n_estimators", config['model']['n_estimators'])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        # Log model (Simpan keseluruhan pipeline agar preprocessing ikut tersimpan)
        mlflow.sklearn.log_model(clf, "model", registered_model_name="Titanic_RF_Model")
        
        print(f"Selesai! Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")

if __name__ == "__main__":
    train_model()