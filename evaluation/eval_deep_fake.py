import pandas as pd
import numpy as np
from eval_metrics import EvalMetrics

def create_synthetic_data(size):
    np.random.seed(10)
    y_pred = np.random.randint(2, size=size)
    y_actual = np.random.randint(2, size=size)
    # print(y_pred)
    # print(y_actual)
    df = pd.DataFrame.from_dict({
            'y_pred': y_pred,
            'y_actual': y_actual,
        })
    return df

df = create_synthetic_data(50)
eval = EvalMetrics(df)
print("accuracy: ", eval.accuracy())
print("precision: ", eval.precision())
print("recall: ", eval.recall())
print("f1_score: ", eval.f1_score())
print("saving confusion matrix file to visualisation folder...")
eval.confusion_matrix()
print("saving auc_roc_curve file to visualisation folder...")
eval.auc_roc_curve()
