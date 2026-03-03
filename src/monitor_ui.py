import pandas as pd
from scipy.stats import ks_2samp
import gradio as gr
from src.config import config

def detect_drift():
    try:
        # Baca data saat model dilatih (reference) dan data dari API (current)
        ref_df = pd.read_csv(config['data']['processed_path'])
        curr_df = pd.read_csv(config['data']['current_path'])
    except FileNotFoundError:
        return "⚠️ File data belum lengkap. Pastikan model sudah ditraining dan API sudah menerima data."

    report = "=== Laporan Continuous Monitoring (Data Drift) ===\n\n"

    # Kita cek fitur numerik utama: Age dan Fare
    fitur_numerik = ['Age', 'Fare']
    
    for col in fitur_numerik:
        if col in ref_df.columns and col in curr_df.columns:
            # Hapus nilai kosong sementara agar statistik tidak error
            data_lama = ref_df[col].dropna()
            data_baru = curr_df[col].dropna()
            
            # Lakukan uji Kolmogorov-Smirnov
            stat, p_value = ks_2samp(data_lama, data_baru)
            
            # Aturan Drift: Jika p-value < 0.05, distribusinya sudah sangat berbeda!
            if p_value < 0.05:
                status = "🚨 DRIFT TERDETEKSI! Model butuh Re-training."
            else:
                status = "✅ Aman. Distribusi data masih stabil."
                
            report += f"Fitur [{col}]:\n"
            report += f" - Status : {status}\n"
            report += f" - P-Value: {p_value:.4f}\n\n"

    return report

# --- Membuat Antarmuka Gradio ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 Dashboard MLOps: Pendeteksi Data Drift")
    gr.Markdown("Dashboard ini memantau apakah data yang diinput oleh user ke API melenceng dari data asli (Titanic 1912).")
    
    with gr.Row():
        btn_cek = gr.Button("Cek Data Drift Sekarang", variant="primary")
    
    output_text = gr.Textbox(label="Hasil Analisis Sistem", lines=10)
    
    # Hubungkan tombol dengan fungsi deteksi
    btn_cek.click(fn=detect_drift, outputs=output_text)

if __name__ == "__main__":
    # Nyalakan server web lokal
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)