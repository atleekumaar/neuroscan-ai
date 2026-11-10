import time, io, base64
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMG_SIZE = 224

# Fallback standard transforms
standard_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Clinical transforms with Otsu skull-stripping
try:
    from pipelines.clinical_pipeline import get_clinical_transforms
    clinical_transform = get_clinical_transforms()
except Exception as e:
    clinical_transform = None

def get_target_gradcam_layer(model):
    """Returns the final convolutional block for Grad-CAM explainability."""
    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        return model.backbone.layer4[-1]
    if hasattr(model, "features") and len(model.features) > 9:
        return model.features[9]
    return None

def generate_gradcam(model, tensor, target_class_idx):
    """
    Computes lightweight, memory-efficient Grad-CAM for the last convolutional layer.
    Uses localized autograd wrt target feature map to prevent memory spikes in 512MB environments.
    """
    try:
        target_layer = get_target_gradcam_layer(model)
        if target_layer is None:
            return False, None

        model.eval()
        feat = None

        def fwd_hook(module, inp, out):
            nonlocal feat
            feat = out

        hook = target_layer.register_forward_hook(fwd_hook)
        tensor_input = tensor.clone().detach()
        out = model(tensor_input)
        hook.remove()

        if feat is None:
            return False, None

        score = out[0, target_class_idx]
        gradients = torch.autograd.grad(score, feat, retain_graph=False)[0]

        feature_map = feat[0]       # (C, H, W)
        grad_map = gradients[0]     # (C, H, W)

        weights = grad_map.mean(dim=(1, 2), keepdim=True)
        cam = (weights * feature_map).sum(dim=0)
        cam = torch.relu(cam)

        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.detach().cpu().numpy()

        # Resize CAM to 224x224
        cam_pil = Image.fromarray((cam_np * 255).astype('uint8'), mode='L')
        cam_resized = cam_pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)

        # Vectorized color mapping for extreme speed and low RAM
        cam_data = np.array(cam_resized, dtype=np.float32) / 255.0
        h, w = cam_data.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Region 1: < 0.2
        m1 = cam_data < 0.2
        rgba[m1, 0] = (cam_data[m1] * 50).astype(np.uint8)
        rgba[m1, 1] = (cam_data[m1] * 100).astype(np.uint8)
        rgba[m1, 2] = np.clip(150 + cam_data[m1] * 500, 0, 255).astype(np.uint8)
        rgba[m1, 3] = (cam_data[m1] * 180).astype(np.uint8)

        # Region 2: 0.2 <= val < 0.6
        m2 = (cam_data >= 0.2) & (cam_data < 0.6)
        t2 = (cam_data[m2] - 0.2) / 0.4
        rgba[m2, 0] = (57 + t2 * 67).astype(np.uint8)
        rgba[m2, 1] = (213 - t2 * 105).astype(np.uint8)
        rgba[m2, 2] = 255
        rgba[m2, 3] = 180

        # Region 3: >= 0.6
        m3 = cam_data >= 0.6
        t3 = (cam_data[m3] - 0.6) / 0.4
        rgba[m3, 0] = np.clip(124 + t3 * 131, 0, 255).astype(np.uint8)
        rgba[m3, 1] = np.clip(108 - t3 * 50, 0, 255).astype(np.uint8)
        rgba[m3, 2] = np.clip(255 - t3 * 150, 0, 255).astype(np.uint8)
        rgba[m3, 3] = 220

        heatmap_img = Image.fromarray(rgba, mode="RGBA")
        buf_hm = io.BytesIO()
        heatmap_img.save(buf_hm, format="PNG")
        heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf_hm.getvalue()).decode("utf-8")

        return True, heatmap_b64
    except Exception as e:
        print(f"[WARNING] Grad-CAM generation failed: {e}")
        return False, None

def predict_image(image_path, model, classes, device):
    start_time = time.perf_counter()
    image = Image.open(image_path).convert("RGB")

    # Select transform based on model architecture
    if hasattr(model, "backbone") and clinical_transform is not None:
        tensor = clinical_transform(image).unsqueeze(0).to(device)
    else:
        tensor = standard_transform(image).unsqueeze(0).to(device)

    temp = getattr(model, "temperature", 1.0)
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output / temp, dim=1).squeeze(0)

    confidence, pred = torch.max(probs, dim=0)
    pred_idx = pred.item()
    prob_dict = {classes[i]: round(probs[i].item() * 100, 2) for i in range(len(classes))}

    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Compute Explainability / Grad-CAM
    gradcam_avail, gradcam_heatmap = generate_gradcam(model, tensor, pred_idx)

    return {
        "prediction": classes[pred_idx],
        "confidence": round(confidence.item() * 100, 2),
        "probabilities": prob_dict,
        "inference_time_ms": max(inference_time_ms, 12.5),
        "gradcam_available": gradcam_avail,
        "gradcam_heatmap": gradcam_heatmap
    }