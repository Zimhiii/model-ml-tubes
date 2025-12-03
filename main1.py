from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import logging
from datetime import datetime
import os
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create directory for saving uploaded images
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- 1. SETUP MODEL (Hanya jalan sekali saat server start) ---
print("Loading Model PyTorch...")
logger.info("Starting model initialization...")

# PERBAIKAN: Definisikan Label sesuai URUTAN TRAINING di Dataset!
# Urutan: ['1000', '10000', '100000', '2000', '20000', '5000', '50000']
CLASS_NAMES = [
    "Seribu Rupiah",        # Index 0 = 1000
    "Sepuluh Ribu Rupiah",  # Index 1 = 10000  
    "Seratus Ribu Rupiah",  # Index 2 = 100000
    "Dua Ribu Rupiah",      # Index 3 = 2000
    "Dua Puluh Ribu Rupiah", # Index 4 = 20000
    "Lima Ribu Rupiah",     # Index 5 = 5000
    "Lima Puluh Ribu Rupiah" # Index 6 = 50000
]

VALUES = [1000, 10000, 100000, 2000, 20000, 5000, 50000]

# Setup Device (Gunakan CPU biar aman di laptop manapun)
device = torch.device("cpu")
logger.info(f"Using device: {device}")

# Load Arsitektur MobileNetV2
model = models.mobilenet_v2(weights=None)
# Ubah layer akhir (classifier) agar outputnya 7 kelas
model.classifier[1] = nn.Linear(model.last_channel, 7)

# Load Bobot dari file .pth
try:
    # state_dict = torch.load("best_model.pth", map_location=device)
    state_dict = torch.load("mobilenetv2_rupiah.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval() # Set mode evaluasi
    print("Model Berhasil Dimuat!")
    logger.info("Model loaded successfully!")
    
    # Log urutan kelas untuk debugging
    logger.info("Class mapping (Index -> Label -> Value):")
    for i, (name, value) in enumerate(zip(CLASS_NAMES, VALUES)):
        logger.info(f"   {i}: {name} -> Rp {value:,}")
        
except Exception as e:
    print(f"Gagal load model: {e}")
    logger.error(f"Failed to load model: {e}")

# Transformasi Gambar (Sama persis dengan training)
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. API ENDPOINT ---
@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {
        "message": "Server Simpenpah Aktif!",
        "class_mapping": {str(i): {"name": name, "value": value} 
                         for i, (name, value) in enumerate(zip(CLASS_NAMES, VALUES))}
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    start_time = datetime.now()
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    saved_filename = f"{timestamp}_{unique_id}{file_extension}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)
    
    logger.info(f"NEW PREDICTION REQUEST - Original: {file.filename}, Saved as: {saved_filename}")
    
    try:
        # Baca gambar yang diupload
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Save the original image
        image.save(saved_path, quality=95)
        logger.info(f"Image saved to: {saved_path} - Dimensions: {image.size}")
        
        # Preprocessing
        tensor = data_transform(image).unsqueeze(0).to(device)
        logger.info(f"Image preprocessed - Tensor shape: {tensor.shape}")
        
        # Prediksi
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Log all class probabilities
            all_probs = probabilities[0].tolist()
            logger.info("All class probabilities:")
            for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, all_probs)):
                logger.info(f"   {i}: {class_name} = {prob*100:.2f}%")
            
            # Ambil nilai tertinggi
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            idx = predicted_idx.item()
            conf_score = confidence.item() * 100
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "success": True,
                "label": CLASS_NAMES[idx],
                "value": VALUES[idx],
                "confidence": round(conf_score, 0),
                "saved_image_path": saved_path,
                "details": {
                    "class_id": idx,
                    "raw_confidence": conf_score,
                    "processing_time_seconds": round(processing_time, 3),
                    "original_filename": file.filename,
                    "saved_filename": saved_filename,
                    "image_dimensions": f"{image.size[0]}x{image.size[1]}",
                    "dataset_class_order": "1000,10000,100000,2000,20000,5000,50000",
                    "all_probabilities": {name: round(prob*100, 2) for name, prob in zip(CLASS_NAMES, all_probs)}
                }
            }
            
            # Log final prediction result
            logger.info("PREDICTION RESULT:")
            logger.info(f"   Predicted: {CLASS_NAMES[idx]} (ID: {idx})")
            logger.info(f"   Value: Rp {VALUES[idx]:,}")
            logger.info(f"   Confidence: {conf_score:.2f}%")
            logger.info(f"   Processing Time: {processing_time:.3f}s")
            logger.info(f"   Original File: {file.filename}")
            logger.info(f"   Saved File: {saved_filename}")
            logger.info("="*50)
            
            return result
             
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        logger.error("PREDICTION FAILED:")
        logger.error(f"   File: {file.filename}")
        logger.error(f"   Error: {error_msg}")
        logger.error(f"   Processing Time: {processing_time:.3f}s")
        logger.error("="*50)
        
        return {"success": False, "error": error_msg}

# Jalankan server dengan: uvicorn main:app --reload --host 0.0.0.0