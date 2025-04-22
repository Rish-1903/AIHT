import os
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def devanagari_preprocess(image):
    """Advanced preprocessing optimized for Devanagari characters"""
    img = np.array(image)
    
    # Check if image needs processing
    if img.mean() < 10 or img.mean() > 245:
        return Image.fromarray(img)
    
    # Apply threshold to separate character from background
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find character contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour (the character)
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Ensure we have a valid character region
        if w > 5 and h > 5:
            # Extract character
            roi = thresh[y:y+h, x:x+w]
            
            # Resize to 28x28 (per dataset spec)
            roi = cv2.resize(roi, (28, 28))
            
            # Add 2px padding as in original dataset
            img = np.pad(roi, ((2,2), (2,2)), mode='constant', constant_values=0)
            
            return Image.fromarray(img)
    
    return Image.fromarray(thresh)

class DevanagariDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Extract clean class names for reporting
        self.class_names = []
        for cls in self.classes:
            parts = cls.split('_')
            if len(parts) >= 3:
                self.class_names.append(parts[2])  # Get the actual character name
            else:
                self.class_names.append(cls)
        
        # Collect all samples
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
