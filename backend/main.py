from pathlib import Path
import os, shutil, uuid
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Support running directly or as module
try:
    from model import load_model
    from predict import predict_image
    from segmentation.router import router as segmentation_router
except ImportError:
    from backend.model import load_model
    from backend.predict import predict_image
    from backend.segmentation.router import router as segmentation_router

BASE_DIR = Path(__file__).resolve().parent.parent
CALIBRATED_MODEL_PATH = BASE_DIR / "models" / "Clinical_Brain_ResNet50_Calibrated.pth"
LEGACY_MODEL_PATH = BASE_DIR / "models" / "NeuroScan_AI_model.pth"
MODEL_PATH = CALIBRATED_MODEL_PATH if CALIBRATED_MODEL_PATH.exists() else LEGACY_MODEL_PATH
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="NEUROSCAN AI — Brain Tumor Classifier & 3D Segmentation API",
    description="Clinical Decision Support System (CDSS) for 2D MRI classification and 3D volumetric tumor segmentation."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, classes = load_model(str(MODEL_PATH), device)
print(f"[INFO] Classification model running on device: {device} with classes: {classes}")

# Mount 3D Volumetric Segmentation API
app.include_router(segmentation_router)

@app.get("/api/status")
async def get_status():
    is_clinical = hasattr(model, "backbone")
    temp = getattr(model, "temperature", 1.0)
    return {
        "status": "online",
        "device": str(device),
        "model_loaded": MODEL_PATH.exists(),
        "model_name": "Clinical Brain ResNet-50 (Calibrated)" if is_clinical else "BrainTumorCNN v1.0",
        "architecture": (
            f"ResNet-50 + Automated Otsu Skull-Stripping + Domain Adaptation + Temperature Scaling (T={temp:.4f})"
            if is_clinical
            else "4x Conv2D (3->32->64->128->256) + MaxPool + Linear(256) + Dropout(0.6) + Linear(4)"
        ),
        "framework": "PyTorch",
        "input_size": "224 x 224 x 3",
        "test_accuracy": "94.67%" if is_clinical else "94.56%",
        "train_accuracy": "99.07%",
        "f1_score": "0.95" if is_clinical else "0.94",
        "meningioma_recall": "97.37%",
        "healthy_recall": "99.38%",
        "pituitary_recall": "99.78%",
        "temperature": temp,
        "classes": classes,
        "segmentation_module_enabled": True
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    temp_path = UPLOAD_DIR / f"{uuid.uuid4()}.{ext}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict_image(str(temp_path), model, classes, device)
    finally:
        if temp_path.exists():
            os.remove(temp_path)

    return result

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")