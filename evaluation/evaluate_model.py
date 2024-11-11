import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

def load_results(filepath):
    """Load the results CSV file."""
    return pd.read_csv(filepath)

def plot_roc_curve_and_determine_threshold(y_true, y_pred_prob, model_name, output_dir):
    """Plot the AUC-ROC curve and determine the optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Find the optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Plot the ROC curve and save
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='black', label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir,f"{model_name}_auc_roc_curve.png"))
    plt.close()

    return optimal_threshold

def plot_confusion_matrix(y_true, y_pred_binary, model_name, output_dir):
    """Plot and save the confusion matrix, 0 = real and 1 = deepfake"""
    cm = confusion_matrix(y_true, y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Deepfake"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir,f"{model_name}_confusion_matrix.png"))
    plt.close()

def save_classification_report_image(y_true, y_pred_binary, model_name, output_dir):
    """Generate and save the classification report as a styled image with formatted values."""
    # Generate the classification report as a DataFrame
    report = classification_report(y_true, y_pred_binary, target_names=["Real", "Deepfake"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.loc['accuracy', 'support'] = None

    # Format to 3 decimal places except for the "support" column
    for column in report_df.columns:
        if column != "support":
            report_df[column] = report_df[column].apply(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
    report_df["support"] = report_df["support"].apply(lambda x: '-' if pd.isna(x) else int(x))

    # Capitalize all row and column labels
    report_df.index = report_df.index.str.title()
    report_df.columns = [col.title() for col in report_df.columns]

    # Start a figure for the table
    fig, ax = plt.subplots(figsize=(12, 4))  # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')

    # Create a color scheme for alternating row colors
    colors = [["#f2f2f2"] * report_df.shape[1] if i % 2 == 0 else ["#ffffff"] * report_df.shape[1] for i in range(report_df.shape[0])]

    # Add the table to the plot with formatting
    table = ax.table(cellText=report_df.values,
                     colLabels=report_df.columns,
                     rowLabels=report_df.index,
                     cellLoc='center',
                     loc='center',
                     cellColours=colors)

    # Styling the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust cell padding

    # Bold headers and row labels
    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:  # Top row or leftmost column
            cell.set_text_props(weight='bold')

    # Save the styled table as an image
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir,f"{model_name}_classification_report.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Styled classification report saved as image to {output_dir}/{model_name}_classification_report.png")

def save_model_diagram(model, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        plot_fpath = os.path.join(output_dir,f"{model_name}_model_diagram.png")
        plot_model(model, to_file=plot_fpath, show_shapes=True, show_layer_names=True)
        print(f"Model diagram saved to {plot_fpath}")
    except Exception as e:
        print(f"An error occurred while plotting model diagram for {model_name} as {e}")

def evaluate_model(model_name, results_path, model_path=None):
    # Load the results CSV file
    results_df = load_results(results_path)
    output_dir = 'evaluation/evaluation-results/'

    # Prepare true labels and predicted probabilities
    y_true = results_df['actual_label']
    y_pred_prob = results_df['predicted_probability']

    # Plot the ROC curve and determine the optimal threshold
    optimal_threshold = plot_roc_curve_and_determine_threshold(y_true, y_pred_prob, model_name, output_dir)

    # Apply the optimal threshold to create binary predictions
    y_pred_binary = [1 if prob >= optimal_threshold else 0 for prob in y_pred_prob]

    # Plot and save the confusion matrix
    plot_confusion_matrix(y_true, y_pred_binary, model_name, output_dir)

    # Save the classification report as an image
    save_classification_report_image(y_true, y_pred_binary, model_name, output_dir)

    # Conditionally load the model and save the model diagram if model_path is provided
    if model_path is not None:
        model = load_model(model_path)
        # model.compile(optimizer='adam', loss='binary_crossentropy')
        save_model_diagram(model, model_name, output_dir)