from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import io
import uvicorn
import os
from datetime import datetime
import uuid

app = FastAPI()

# ==========================================
# 0. SETUP FOLDER UPLOAD
# ==========================================

UPLOAD_FOLDER = "uploaded_images"

# Buat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"📁 Folder upload: {os.path.abspath(UPLOAD_FOLDER)}")

def save_uploaded_image(image: Image.Image, original_filename: str, prediction_info: dict) -> str:
    """
    Simpan gambar yang diupload dengan nama unik
    """
    try:
        # Generate nama file unik dengan timestamp dan UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Ambil ekstensi file asli, default ke .jpg
        file_ext = os.path.splitext(original_filename)[1].lower()
        if not file_ext or file_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            file_ext = '.jpg'
        
        # Format nama: timestamp_prediksi_confidence_uniqueid.ext
        prediction_label = prediction_info['label'].replace(' ', '_').replace('Rupiah', 'Rp')
        confidence = prediction_info['confidence']
        
        filename = f"{timestamp}_{prediction_label}_{confidence}pct_{unique_id}{file_ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Simpan gambar
        # Convert ke RGB untuk memastikan kompatibilitas
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(filepath, 'JPEG', quality=95)
        
        print(f"💾 Gambar disimpan: {filename}")
        return filepath
        
    except Exception as e:
        print(f"❌ Error saat menyimpan gambar: {e}")
        return None

# ==========================================
# 1. SETUP MODEL & LOGIKA (SAMA PERSIS COLAB)
# ==========================================

print("⏳ Memuat Model PyTorch...")

# URUTAN KELAS (Alfabetikal sesuai training)
# 0: 1000, 1: 10000, 2: 100000, 3: 2000, 4: 20000, 5: 5000, 6: 50000
CLASS_INFO = [
    {"label": "Seribu Rupiah", "value": 1000},          # Index 0 ('1000')
    {"label": "Sepuluh Ribu Rupiah", "value": 10000},   # Index 1 ('10000')
    {"label": "Seratus Ribu Rupiah", "value": 100000},  # Index 2 ('100000')
    {"label": "Dua Ribu Rupiah", "value": 2000},        # Index 3 ('2000')
    {"label": "Dua Puluh Ribu Rupiah", "value": 20000}, # Index 4 ('20000')
    {"label": "Lima Ribu Rupiah", "value": 5000},       # Index 5 ('5000')
    {"label": "Lima Puluh Ribu Rupiah", "value": 50000} # Index 6 ('50000')
]

# Setup Device (CPU aman untuk laptop/server standar)
device = torch.device("cpu")

# Definisi Arsitektur (MobileNetV2)
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 7) # 7 Kelas

# Load Bobot
MODEL_PATH = "mobilenetv2_rupiah.pth" # GANTI dengan nama file .pth kamu yg asli
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() # Mode Evaluasi (PENTING!)
    print(f"✅ Model '{MODEL_PATH}' Berhasil Dimuat!")
except Exception as e:
    print(f"❌ ERROR LOAD MODEL: {e}")
    print("Pastikan nama file .pth benar dan ada di folder ini.")

# Setup Transformasi (Resep Masak - Sama persis Colab)
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalisasi ImageNet (Wajib untuk MobileNetV2)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 2. API ENDPOINTS
# ==========================================

@app.get("/")
def root():
    # Hitung jumlah gambar yang tersimpan
    image_count = len([f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    return {
        "status": "Server Simpenpah Online", 
        "model": MODEL_PATH,
        "upload_folder": UPLOAD_FOLDER,
        "saved_images": image_count
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Baca File Gambar
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # 2. FIX ROTASI (PENTING: Ini rahasia kenapa Colab bagus tapi HP jelek)
        # HP sering kirim gambar miring, ini memutarnya sesuai EXIF
        img = ImageOps.exif_transpose(img)
        
        # Convert ke RGB (jaga-jaga kalo format PNG/RGBA)
        img = img.convert('RGB')

        # 3. Preprocessing (Transform)
        input_tensor = data_transform(img).unsqueeze(0).to(device)

        # 4. Prediksi (Inference)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Ambil nilai tertinggi
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            idx = predicted_idx.item()     # Index kelas (0-6)
            score = confidence.item() * 100 # Persentase (0-100)
            
            result_data = CLASS_INFO[idx]

            # Prepare prediction info untuk save
            prediction_info = {
                "label": result_data["label"],
                "value": result_data["value"],
                "confidence": round(score, 0)
            }

            # 5. SIMPAN GAMBAR (NEW!)
            saved_path = save_uploaded_image(img, file.filename or "unknown.jpg", prediction_info)

            # Debugging di Terminal Laptop
            print(f"📸 Scan Masuk -> Prediksi: {result_data['label']} ({score:.2f}%)")
            if saved_path:
                print(f"📁 File tersimpan di: {saved_path}")

            return {
                "success": True,
                "label": result_data["label"],
                "value": result_data["value"],
                "confidence": round(score, 0),
                "saved_path": saved_path,
                "original_filename": file.filename
            }

    except Exception as e:
        print(f"❌ Error saat prediksi: {e}")
        return {"success": False, "error": str(e)}

@app.get("/images/count")
def get_image_count():
    """Endpoint untuk cek berapa gambar yang tersimpan"""
    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        return {
            "success": True,
            "count": len(files),
            "folder": UPLOAD_FOLDER,
            "recent_files": sorted(files, reverse=True)[:10]  # 10 file terbaru
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Cara Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000