import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model


def load_results(filepath):
    """Load the results CSV file."""
    return pd.read_csv(filepath)

def plot_confusion_matrix(y_true, y_pred_binary, output_path='results/confusion_matrix.png'):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Deepfake"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

def save_classification_report(y_true, y_pred_binary, output_path='results/classification_report.txt'):
    """Generate and save the classification report."""
    report = classification_report(y_true, y_pred_binary, target_names=["Real", "Deepfake"])
    with open(output_path, 'w') as file:
        file.write(report)
    print("Classification report saved to", output_path)

def plot_roc_curve(y_true, y_pred_prob, output_path='results/auc_roc.png'):
    """Plot and save the AUC-ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for random chance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

def save_model_diagram(model, output_path='results/model_diagram.png'):
    """Generate and save the model diagram."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_model(model, to_file=output_path, show_shapes=True, show_layer_names=True)
    print("Model diagram saved to", output_path)

if __name__ == '__main__':
    # Load the results CSV file
    results_df = load_results('results/video_classification_results.csv')

    # Prepare true labels and predicted probabilities
    y_true = results_df['actual_label']
    y_pred_prob = results_df['predicted_probability']

    # Apply threshold to create binary predictions
    threshold = 0.5
    y_pred_binary = [1 if prob >= threshold else 0 for prob in y_pred_prob]

    # Plot and save the confusion matrix
    plot_confusion_matrix(y_true, y_pred_binary)

    # Save the classification report to a file
    save_classification_report(y_true, y_pred_binary)

    # Plot and save the AUC-ROC curve
    plot_roc_curve(y_true, y_pred_prob)

    # Plot and save model diagram
    best_model_loaded = load_model("results/best_cnn_model.h5")

    # Save the model diagram
    save_model_diagram(best_model_loaded)