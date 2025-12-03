import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 7)
state_dict = torch.load("best_model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

CLASS_NAMES = [
    "Seribu Rupiah",        # Index 0 = 1000
    "Sepuluh Ribu Rupiah",  # Index 1 = 10000  
    "Seratus Ribu Rupiah",  # Index 2 = 100000
    "Dua Ribu Rupiah",      # Index 3 = 2000
    "Dua Puluh Ribu Rupiah", # Index 4 = 20000
    "Lima Ribu Rupiah",     # Index 5 = 5000
    "Lima Puluh Ribu Rupiah" # Index 6 = 50000
]

# Transform yang sama
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def debug_single_image(image_path):
    """Debug prediksi untuk satu gambar"""
    print(f"\n=== DEBUGGING: {image_path} ===")
    
    image = Image.open(image_path).convert('RGB')
    print(f"Original size: {image.size}")
    
    tensor = data_transform(image).unsqueeze(0)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor min/max: {tensor.min():.3f} / {tensor.max():.3f}")
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        print("\nAll probabilities:")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities[0])):
            print(f"  {i}: {name:<20} = {prob*100:6.2f}%")
        
        # Cek apakah semua probabilitas hampir sama (tanda model bingung)
        probs = probabilities[0].numpy()
        if np.std(probs) < 0.1:
            print("⚠️  WARNING: Semua probabilitas hampir sama! Model mungkin tidak terlatih dengan baik.")
        
        predicted = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted].item() * 100
        
        print(f"\nPREDICTION: {CLASS_NAMES[predicted]} ({confidence:.1f}%)")

# Test beberapa gambar dari folder uploaded_images
# if os.path.exists("uploaded_images"):
#     images = [f for f in os.listdir("uploaded_images") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     for img in images[:5]:  # Test 5 gambar pertama
#         debug_single_image(os.path.join("uploaded_images", img))
if os.path.exists("wa"):
    images = [f for f in os.listdir("wa") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img in images[:5]:  # Test 5 gambar pertama
        debug_single_image(os.path.join("wa", img))