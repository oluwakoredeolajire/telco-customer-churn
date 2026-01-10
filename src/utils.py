import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def save_evaluation_metrics(metrics, filepath):
    """Save evaluation metrics to a JSON file."""
    
    # Check if the file exists and is non-empty
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            with open(filepath, 'r') as file:
                # Load existing metrics
                existing_metrics = json.load(file)
        except json.JSONDecodeError:
            # If the file is corrupted or empty, initialize a new dict
            print(f"Warning: {filepath} is corrupted or empty. Initializing new file.")
            existing_metrics = {}
    else:
        existing_metrics = {}  # Initialize an empty dict if file doesn't exist or is empty

    # Update metrics with the new evaluation results
    model_type = metrics.get('model_type')  # Ensure model_type is included in metrics
    if model_type:
        existing_metrics[model_type] = metrics
    else:
        raise ValueError("Model type is missing in the metrics.")

    # Save the updated metrics back to the file
    with open(filepath, 'w') as file:
        json.dump(existing_metrics, file, indent=4)

    print(f"Metrics saved to {filepath}")

def plot_roc_curve(y_true, y_scores, model_type, save_path):
    """Plot ROC curve and save the figure."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_type}')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")

def plot_confusion_matrix(cm, model_type, save_path):
    """Plot confusion matrix and save the figure."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_type}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

