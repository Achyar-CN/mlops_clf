import yaml
from pathlib import Path

# Mendapatkan path absolut ke params.yaml agar tidak error saat dipanggil dari folder lain
config_path = Path(__file__).parent / "params.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)