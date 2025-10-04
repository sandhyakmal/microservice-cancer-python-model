from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os

# Konfigurasi dasar aplikasi
app = FastAPI(
    title="API Deteksi Kanker Payudara",
    version="1.0.0",
    root_path="/models"
)

# Daftar kelas dan path model
DAFTAR_KELAS = ["Cancer", "Non-Cancer"]
LOKASI_MODEL = "breast_cancer_model.h5"
model = None

# Nonaktifkan log TensorFlow yang tidak penting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@app.on_event("startup")
async def muat_model_saat_startup():
    """Memuat model saat aplikasi dijalankan"""
    global model
    if not os.path.exists(LOKASI_MODEL):
        raise RuntimeError(f"Berkas model tidak ditemukan: {LOKASI_MODEL}")
    model = keras.models.load_model(LOKASI_MODEL, compile=False)
    print("Model berhasil dimuat")


def praproses_gambar(gambar: Image.Image) -> np.ndarray:
    """Melakukan praproses pada gambar sebelum diprediksi"""
    gambar = gambar.convert('RGB').resize((224, 224))
    array_gambar = np.expand_dims(np.array(gambar, dtype=np.float32) / 255.0, axis=0)
    return array_gambar


@app.get("/")
async def root():
    """Endpoint utama untuk pengecekan status API"""
    return {
        "pesan": "API Deteksi Kanker Payudara sedang berjalan",
        "model_dimuat": model is not None,
        "versi_tensorflow": tf.__version__,
        "versi_keras": keras.__version__,
        "root_path": app.root_path
    }


@app.post("/predict")
async def prediksi(file: UploadFile = File(...)):
    """Melakukan prediksi dari gambar yang diunggah"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model belum dimuat")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Berkas harus berupa gambar")

    try:
        gambar = Image.open(file.file)
        data_gambar = praproses_gambar(gambar)
        hasil_prediksi = model.predict(data_gambar, verbose=0)[0]

        indeks_kelas = int(np.argmax(hasil_prediksi))
        return {
            "predicted_class": DAFTAR_KELAS[indeks_kelas],
            "confidence": float(np.max(hasil_prediksi)),
            "probabilities": {
                DAFTAR_KELAS[i]: float(hasil_prediksi[i]) for i in range(len(DAFTAR_KELAS))
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat prediksi: {str(e)}")


@app.post("/reload-model")
async def muat_ulang_model():
    """Memuat ulang model secara manual"""
    global model
    model = keras.models.load_model(LOKASI_MODEL, compile=False)
    return {"pesan": "Model berhasil dimuat ulang"}
