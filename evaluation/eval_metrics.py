"""
Given 2 columns (y_pred, y_actual),
provide evaluation metrics such as f1, precision,recall, confusion metrics, auc-roc curve etc
and save visualisations files
"""
import os
import matplotlib.pyplot as plt
from sklearn import metrics

class EvalMetricsAbstract:
    def accuracy(self):
        raise NotImplementedError
    
    def precision(self):
        raise NotImplementedError
    
    def recall(self):
        raise NotImplementedError
    
    def f1_score(self):
        raise NotImplementedError
    
    def confusion_matrix(self):
        raise NotImplementedError
    
    def auc_roc_curve(self):
        raise NotImplementedError
    

class EvalMetrics(EvalMetricsAbstract):
    def __init__(self, df) -> None:
        """
        Initialize with a dataframe containing two columns: y_pred and y_actual.
        """
        self.df = df
        self.y_pred = df['y_pred']
        self.y_actual = df['y_actual']

        # Create visualisations folder if it doesn't exist
        self.visualisations_path = os.path.join(os.path.dirname(__file__), 'visualisations')
        if not os.path.exists(self.visualisations_path):
            os.makedirs(self.visualisations_path)

    def accuracy(self):
        """
        return number of correct predictions / total predictions
        """
        return metrics.accuracy_score(self.y_actual, self.y_pred)
    
    def precision(self):
        """
        return TP / (TP+FP)
        """
        return metrics.precision_score(self.y_actual, self.y_pred)
    
    def recall(self):
        """
        return TP / (TP+FN)
        """
        return metrics.recall_score(self.y_actual, self.y_pred)
    
    def f1_score(self):
        """
        return 2*precision*recall / (precision+recall)
        """
        return metrics.f1_score(self.y_actual, self.y_pred)
    
    def confusion_matrix(self):
        confusion_matrix = metrics.confusion_matrix(self.y_actual, self.y_pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
        cm_display.plot()
        plt.title("Confusion Matrix")
        # Save the confusion matrix plot
        save_path = os.path.join(self.visualisations_path, 'confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()
    
    def auc_roc_curve(self):
        fpr, tpr, _ = metrics.roc_curve(self.y_actual, self.y_pred)
        auc = metrics.roc_auc_score(self.y_actual, self.y_pred)
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='r')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        # Save the AUC-ROC curve plot
        save_path = os.path.join(self.visualisations_path, 'auc_roc_curve.png')
        plt.savefig(save_path)
        plt.close()