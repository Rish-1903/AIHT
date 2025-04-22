from torchvision import transforms
from .dataset import devanagari_preprocess

def get_train_transform():
    return transforms.Compose([
        transforms.Lambda(devanagari_preprocess),
        transforms.Resize((64, 64)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Lambda(devanagari_preprocess),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
