## Evaluation Metrics for Binary Classification

This folder provides a set of tools for evaluating binary classification models, including calculating common metrics and visualizing results. It supports metrics like accuracy, precision, recall, F1-score, and generates visualizations of the confusion matrix and AUC-ROC curve.

### Usage

#### Create Synthetic Data

The `create_synthetic_data` function generates random binary predictions and labels for testing the evaluation metrics.

#### Calculate Evaluation Metrics

The `EvalMetrics` class provides methods for calculating the following metrics:

- **Accuracy**: The ratio of correct predictions to the total predictions.
- **Precision**: The ratio of true positives to the sum of true positives and false positives.
- **Recall**: The ratio of true positives to the sum of true positives and false negatives.
- **F1 Score**: The harmonic mean of precision and recall.

#### Generate Visualizations

The `EvalMetrics` class also generates and saves two types of plots:

- **Confusion Matrix**: Displays the counts of true positives, false positives, true negatives, and false negatives. Saved as `confusion_matrix.png` in the `visualisations` folder.
- **AUC-ROC Curve**: Shows the trade-off between true positive rate and false positive rate at different thresholds. The Area Under the Curve (AUC) score is also shown. Saved as `auc_roc_curve.png` in the `visualisations` folder.
