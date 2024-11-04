import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc

def plot_confusion_matrix(results_df, class_labels=["deepfake", "real"]):
    labels = class_labels
    cm = confusion_matrix(results_df["Actual"], results_df["Predicted"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)  
    disp.plot(cmap="Blues") 
    plt.title("Confusion Matrix")
    plt.show()
    report = classification_report(results_df["Actual"], results_df["Predicted"], target_names=labels)
    print(report)

df = pd.read_csv("results.csv")


def plot_auc_roc(df):
    fpr, tpr, threshold = roc_curve(df["Actual"], df["Predicted"])
    roc_auc = auc(fpr, tpr)

    plt.title('AUC-ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.show()

plot_auc_roc(df)
plot_confusion_matrix(df)