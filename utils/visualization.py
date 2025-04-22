import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import torch
import torch.nn.functional as F

def plot_class_accuracy(all_labels, all_preds, class_names):
    class_correct = np.zeros(len(class_names))
    class_total = np.zeros(len(class_names))
    
    for i in range(len(class_names)):
        class_mask = (all_labels == i)
        class_correct[i] = (all_preds[class_mask] == all_labels[class_mask]).sum()
        class_total[i] = class_mask.sum()
    
    class_acc = (class_correct / class_total) * 100
    
    plt.figure(figsize=(15, 8))
    sns.barplot(x=class_names, y=class_acc)
    plt.title('Class-wise Accuracy (%)')
    plt.xlabel('Character Class')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.ylim(0, 100)
    plt.tight_layout()
    return plt

def plot_pr_curve(all_labels, all_probs, class_names):
    y_test_bin = label_binarize(all_labels, classes=np.arange(len(class_names)))
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(len(class_names)):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i])
        average_precision[i] = auc(recall[i], precision[i])
    
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test_bin.ravel(), all_probs.ravel())
    average_precision["micro"] = auc(recall["micro"], precision["micro"])
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall["micro"], precision["micro"],
             label=f'micro-average Precision-Recall (AP = {average_precision["micro"]:0.2f})')
    
    for i in range(min(5, len(class_names))):
        plt.plot(recall[i], precision[i],
                 label=f'{class_names[i]} (AP = {average_precision[i]:0.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid()
    return plt

# ... (other visualization functions)
