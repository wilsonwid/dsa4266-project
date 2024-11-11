import os
from evaluate_model import evaluate_model

# List of model and result paths for evaluation
evaluations = [
    {"model_name": "cnn", "model_path": "../models/cnn/results/best_cnn_model.h5", "results_path": "../models/cnn/results/video_classification_results.csv"}, # CNN
    {"model_name": "resnet", "model_path": "path/to/second_model.h5", "results_path": "path/to/second_results.csv"}, # ResNet
    {"model_name": "videomae", "model_path": None, "results_path": "path/to/second_results.csv"}, # VideoMAE (no model path)
]

# Loop through each model and results pair, running evaluation for each
for eval_info in evaluations:
    model_path = eval_info["model_path"]
    results_path = eval_info["results_path"]
    model_name = eval_info["model_name"]

    # Check if both paths exist (model_path check is skipped if it's None)
    if model_path and not os.path.exists(model_path):
        print(f"Model file not found: {model_path}. Skipping this evaluation.")
        continue
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}. Skipping this evaluation.")
        continue

    print(f"Running evaluation for model: {model_name} with results: {results_path}")
    try:
        # Pass None if model_path is not provided
        evaluate_model(model_name, results_path, model_path if model_path else None)
    except Exception as e:
        print(f"An error occurred while evaluating model {model_name} with results {results_path}: {e}")