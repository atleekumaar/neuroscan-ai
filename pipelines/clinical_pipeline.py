import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class SkullStripper:
    """Automated Otsu Skull-Stripping to eliminate shortcut learning."""
    @staticmethod
    def strip_and_crop(pil_img):
        img_np = np.array(pil_img.convert('L'))
        blurred = cv2.GaussianBlur(img_np, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(img_np)
            cv2.drawContours(mask, [largest_cnt], -1, 255, thickness=-1)
            brain_only = cv2.bitwise_and(img_np, img_np, mask=mask)
            x, y, w, h = cv2.boundingRect(largest_cnt)
            crop = brain_only[y:y+h, x:x+w]
        else:
            crop = img_np
            
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(crop_rgb)

class ClinicalFocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def get_clinical_transforms():
    return transforms.Compose([
        transforms.Lambda(SkullStripper.strip_and_crop),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def build_model(num_classes=4):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
