import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from ..models.ehcrnetv2 import EHCRNetV2
from ..utils.dataset import DevanagariDataset
from ..utils.transforms import get_train_transform, get_test_transform

def train_model(dataset_path, device='cuda', batch_size=64, epochs=50):
    # Setup datasets
    train_dir = os.path.join(dataset_path, "Train")
    test_dir = os.path.join(dataset_path, "Test")
    
    train_dataset = DevanagariDataset(train_dir, transform=get_train_transform())
    test_dataset = DevanagariDataset(test_dir, transform=get_test_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    
    # Initialize model
    model = EHCRNetV2(num_classes=len(train_dataset.classes)).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=epochs)
    
    # Training loop
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Accuracy: {acc:.2f}%")
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")
    
    return model

if __name__ == "__main__":
    dataset_path = "data/DevanagariHandwrittenCharacterDataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_model(dataset_path, device=device)
