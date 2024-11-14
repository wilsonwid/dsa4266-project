from models.rcnn.results.transform_rcnn_results import transform_rcnn_predictions


if __name__ == "__main__":
    path_to_results = "models/cnn_lstm/results/predictions_cnn_encoder_lstm.csv"
    path_to_output = "models/cnn_lstm/results/predictions_cnn_encoder_lstm_transformed.csv"
    transform_rcnn_predictions(path_to_results, path_to_output)