import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from ..models.ehcrnetv2 import EHCRNetV2
from ..utils.dataset import DevanagariDataset
from ..utils.transforms import get_test_transform
from ..utils.visualization import plot_class_accuracy, plot_pr_curve

def evaluate_model(model_path, dataset_path, device='cuda'):
    # Load data
    test_dataset = DevanagariDataset(
        os.path.join(dataset_path, "Test"),
        transform=get_test_transform()
    )
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)
    
    # Load model
    model = EHCRNetV2(num_classes=len(test_dataset.classes)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Generate reports
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.class_names))
    
    # Generate visualizations
    plot_class_accuracy(all_labels, all_preds, test_dataset.class_names).show()
    plot_pr_curve(all_labels, all_probs, test_dataset.class_names).show()
    
    return {
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs,
        'class_names': test_dataset.class_names
    }

if __name__ == "__main__":
    results = evaluate_model(
        model_path="best_model.pth",
        dataset_path="data/DevanagariHandwrittenCharacterDataset"
    )
