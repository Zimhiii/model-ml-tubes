from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import io
import os
import time

app = FastAPI()

# ==========================================
# 1. KONFIGURASI PENYIMPANAN GAMBAR
# ==========================================
UPLOAD_DIR = "uploaded_images"

# Buat folder jika belum ada
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    print(f"📂 Folder '{UPLOAD_DIR}' berhasil dibuat.")

# Opsional: Jika ingin gambar bisa diakses lewat URL (http://ip:8000/images/...)
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

# ==========================================
# 2. SETUP MODEL AI (Sesuai Notebook)
# ==========================================
print("⏳ Memuat Model PyTorch...")

# URUTAN KELAS DARI NOTEBOOK (Alfabetikal String)
# ['1000', '10000', '100000', '2000', '20000', '5000', '50000']
CLASS_INFO = [
    {"label": "Seribu Rupiah", "value": 1000},          # Index 0: '1000'
    {"label": "Sepuluh Ribu Rupiah", "value": 10000},   # Index 1: '10000'
    {"label": "Seratus Ribu Rupiah", "value": 100000},  # Index 2: '100000'
    {"label": "Dua Ribu Rupiah", "value": 2000},        # Index 3: '2000'
    {"label": "Dua Puluh Ribu Rupiah", "value": 20000}, # Index 4: '20000'
    {"label": "Lima Ribu Rupiah", "value": 5000},       # Index 5: '5000'
    {"label": "Lima Puluh Ribu Rupiah", "value": 50000} # Index 6: '50000'
]

device = torch.device("cpu") # Gunakan CPU biar aman di semua laptop

# Definisi Arsitektur MobileNetV2
model = models.mobilenet_v2(weights=None)
# Ubah layer classifier terakhir agar outputnya 7 kelas
model.classifier[1] = nn.Linear(model.last_channel, 7)

# Model file
MODEL_FILENAME = "best_model.pth" 

try:
    if os.path.exists(MODEL_FILENAME):
        state_dict = torch.load(MODEL_FILENAME, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"✅ Model '{MODEL_FILENAME}' Berhasil Dimuat!")
    else:
        print(f"⚠️ File model '{MODEL_FILENAME}' tidak ditemukan! Pastikan nama file benar.")
except Exception as e:
    print(f"❌ Error Load Model: {e}")

# Transformasi Gambar (Sesuai training di Notebook)
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. API ENDPOINTS
# ==========================================

@app.get("/")
def root():
    return {"status": "Server Simpenpah Ready", "upload_folder": UPLOAD_DIR}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # --- A. BACA FILE ---
        image_bytes = await file.read()
        
        # --- B. SIMPAN GAMBAR KE FOLDER ---
        # Buat nama file unik pakai timestamp biar gak ketimpa
        filename = f"scan_{int(time.time())}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Tulis file ke disk
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        
        print(f"💾 Gambar disimpan: {file_path}")

        # --- C. PREPROCESSING ---
        img = Image.open(io.BytesIO(image_bytes))
        
        # Fix Rotasi HP (Wajib biar akurat)
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')
        
        # Apply transforms
        input_tensor = data_transform(img).unsqueeze(0).to(device)

        # --- D. PREDIKSI ---
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Ambil nilai tertinggi
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            idx = predicted_idx.item()     # Index kelas (0-6)
            score = confidence.item() * 100 # Persentase (0-100)
            
            result_data = CLASS_INFO[idx]

            # Debugging di Terminal
            print(f"📸 Prediksi: {result_data['label']} ({score:.1f}%)")

            return {
                "success": True,
                "label": result_data["label"],
                "value": result_data["value"],
                "confidence": round(score, 1),
                "saved_as": filename
            }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

# Jalankan dengan: uvicorn main:app --reload --host 0.0.0.0 --port 8000