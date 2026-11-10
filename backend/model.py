import torch
import torch.nn as nn
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_model(model_path, device):
    """
    Loads model weights for inference. Automatically detects whether the checkpoint
    is the calibrated Clinical ResNet-50 architecture or the legacy BrainTumorCNN.
    """
    DEFAULT_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
    DEFAULT_TEMPERATURE = 1.297
    try:
        # PyTorch 2.6+ compatibility
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)

        # 1. Check if checkpoint is Clinical ResNet-50
        if isinstance(checkpoint, dict) and ("model_state" in checkpoint or "temperature" in checkpoint):
            from pipelines.clinical_pipeline import ClinicalResNet50
            temperature = float(checkpoint.get("temperature", DEFAULT_TEMPERATURE))
            state_dict = checkpoint["model_state"]
            model = ClinicalResNet50(num_classes=len(DEFAULT_CLASSES)).to(device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            model.temperature = temperature
            print(f"[INFO] Loaded calibrated Clinical ResNet-50 weights from {model_path} (T={temperature:.4f})")
            return model, DEFAULT_CLASSES

        # 2. Check if state_dict has backbone prefix (ResNet-50 without temperature key)
        raw_state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(raw_state, dict) and any(k.startswith("backbone.") for k in raw_state.keys()):
            from pipelines.clinical_pipeline import ClinicalResNet50
            model = ClinicalResNet50(num_classes=len(DEFAULT_CLASSES)).to(device)
            model.load_state_dict(raw_state, strict=True)
            model.eval()
            model.temperature = DEFAULT_TEMPERATURE
            print(f"[INFO] Loaded Clinical ResNet-50 backbone weights from {model_path}")
            return model, DEFAULT_CLASSES

        # 3. Fallback: Legacy BrainTumorCNN
        classes = checkpoint.get("classes", DEFAULT_CLASSES) if isinstance(checkpoint, dict) else DEFAULT_CLASSES
        model = BrainTumorCNN(num_classes=len(classes)).to(device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.temperature = 1.0
        print(f"[INFO] Loaded legacy BrainTumorCNN weights from {model_path}")
        return model, classes

    except Exception as e:
        print(f"[WARNING] Could not load model weights from {model_path}: {e}")
        try:
            from pipelines.clinical_pipeline import ClinicalResNet50
            print("[INFO] Initializing default ClinicalResNet50 architecture (Untrained mode).")
            model = ClinicalResNet50(num_classes=len(DEFAULT_CLASSES)).to(device)
        except Exception:
            print("[INFO] Initializing default BrainTumorCNN architecture (Untrained/Demo mode).")
            model = BrainTumorCNN(num_classes=len(DEFAULT_CLASSES)).to(device)
        model.eval()
        model.temperature = 1.0
        return model, DEFAULT_CLASSES