import yaml
from pathlib import Path
from src.config import config

def test_config_structure():
    # Memastikan file params.yaml memiliki struktur yang wajib ada
    assert 'data' in config
    assert 'model' in config
    assert 'features' in config['model']
    assert 'target_col' in config['model']

def test_feature_consistency():
    # Memastikan tidak ada fitur duplikat di konfigurasi
    features = config['model']['features']
    assert len(features) == len(set(features)), "Ada fitur duplikat di params.yaml!"
    
    # Memastikan target tidak termasuk di dalam daftar fitur
    assert config['model']['target_col'] not in features, "Target column tidak boleh ada di daftar features!"