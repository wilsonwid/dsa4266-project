'''
1) RCNN output is not a probability, rather is a 2-tuple after softmax containing probabilities
2) argmax  is the "predicted" column
3) and the "probability" is the max value in the 2-tuple

Hence, we do some transformation on the csv.
if “predicted” = 0, predicted_probability = 1 - probability
else predicted_probability = probability
This is to ensure that “probability” will always refer to probability of it being a deepfake
'''
import pandas as pd

def transform_rcnn_predictions(path_to_results, path_to_output):
    df = pd.read_csv(path_to_results)
    df['actual_label'] = df['actual']
    df['predicted_probability'] = df['probability'] * (df['predicted'] == 1) + (1 - df['probability']) * (df['predicted'] == 0)
    df = df[['actual_label', 'predicted_probability']]
    df.to_csv(path_to_output)

if __name__ == "__main__":
    path_to_results = "models/rcnn/results/predictions_rcnn.csv"
    path_to_output = "models/rcnn/results/predictions_rcnn_transformed.csv"
    transform_rcnn_predictions(path_to_results, path_to_output)
    