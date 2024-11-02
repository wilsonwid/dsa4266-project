import os
import cv2
import json
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, TimeDistributed
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras_tuner.tuners import RandomSearch
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import CSVLogger

# Load the results CSV file
results_df = pd.read_csv('results/video_classification_results.csv')

# Prepare the true labels and predicted probabilities for confusion matrix and classification report
y_true = results_df['actual_label']
y_pred_prob = results_df['predicted_probability']

# Apply threshold to create binary predictions (adjust the threshold if needed)
threshold = 0.5
y_pred_binary = [1 if prob >= threshold else 0 for prob in y_pred_prob]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Deepfake"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
os.makedirs('results', exist_ok=True)
plt.savefig('results/confusion_matrix.png')
plt.show()

# Print classification report
print(classification_report(y_true, y_pred_binary, target_names=["Real", "Deepfake"]))

# Plot AUC-ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
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
os.makedirs('results', exist_ok=True)
plt.savefig('results/auc_roc.png')
plt.show()