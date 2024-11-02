import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Check GPU
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model

def create_datasets(base_dir="../../data"):
    # Separate lists for each dataset
    train_data = []
    val_data = []
    test_data = []

    # Define the folders and subfolders to iterate through
    for folder in ['train', 'validation', 'test']:
        for subfolder in ['real', 'deepfake']:
            subfolder_path = os.path.join(base_dir, folder, subfolder)

            # Check if the subfolder exists
            if not os.path.exists(subfolder_path):
                print(f"Subfolder {subfolder_path} does not exist.")
                continue

            # Loop through each subdirectory (representing each video) in the subfolder
            for video_dir in os.listdir(subfolder_path):
                video_path = os.path.join(subfolder_path, video_dir)

                # Ensure the path is a directory (each video is in a subdirectory)
                if os.path.isdir(video_path):
                    # Create a label based on the subfolder
                    label = 1 if subfolder == 'deepfake' else 0

                    # Loop through each frame in the video directory
                    for frame_file in os.listdir(video_path):
                        frame_path = os.path.join(video_path, frame_file)

                        # Check if the file is an image
                        if frame_file.endswith(('.jpg', '.png', '.jpeg')) and os.path.isfile(frame_path):
                            # Append the frame path and label to the appropriate dataset list
                            if folder == 'train':
                                train_data.append({'file_path': frame_path, 'label': label})
                            elif folder == 'validation':
                                val_data.append({'file_path': frame_path, 'label': label})
                            elif folder == 'test':
                                test_data.append({'file_path': frame_path, 'label': label})

    # Create DataFrames from the collected data
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    # Shuffle the DataFrames
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return train_df, val_df, test_df

def load_and_preprocess_image(file_path):
    # Load the image from the file path
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Decode the image
    image = tf.image.resize(image, [224, 224])  # Resize to target size
    image = image / 255.0  # Normalize to [0, 1]
    return image

def predict_videos(model, test_df, batch_size=64):
    # Dictionary to store video predictions and labels
    video_predictions = {}
    all_predictions = []
    all_true_labels = []

    # Lists to store images and labels in each batch
    batch_images = []
    batch_true_labels = []
    video_dirs = []

    # Counter to track number of predictions
    prediction_count = 0

    for index, row in test_df.iterrows():
        frame_path = row['file_path']
        video_dir = os.path.dirname(frame_path)  # Get the directory of the video
        true_label = row['label']  # Retrieve the true label for this frame

        # Load and preprocess the image
        image = load_and_preprocess_image(frame_path)
        batch_images.append(image)
        batch_true_labels.append(true_label)
        video_dirs.append(video_dir)

        # Once batch is filled, predict and process results
        if len(batch_images) == batch_size:
            # Stack images in the batch along the first dimension
            batch_images = np.stack(batch_images)
            # Predict on the batch
            batch_predictions = model.predict(batch_images, verbose=0).flatten()

            # Process predictions
            for i, prediction_value in enumerate(batch_predictions):
                video_dir = video_dirs[i]
                true_label = batch_true_labels[i]

                if video_dir not in video_predictions:
                    video_predictions[video_dir] = {
                        'predictions': [],
                        'true_labels': true_label  # Store the true label
                    }
                video_predictions[video_dir]['predictions'].append(prediction_value)
                all_predictions.append(prediction_value)
                all_true_labels.append(true_label)
                prediction_count += 1

            # Print progress every 10,000 predictions
            if prediction_count % 10000 == 0:
                print(f"{prediction_count} predictions processed.")

            # Clear batch data for next set
            batch_images, batch_true_labels, video_dirs = [], [], []

    # Predict any remaining images that didn’t fill the last batch
    if batch_images:
        batch_images = np.stack(batch_images)
        batch_predictions = model.predict(batch_images, verbose=0).flatten()
        for i, prediction_value in enumerate(batch_predictions):
            video_dir = video_dirs[i]
            true_label = batch_true_labels[i]
            if video_dir not in video_predictions:
                video_predictions[video_dir] = {
                    'predictions': [],
                    'true_labels': true_label
                }
            video_predictions[video_dir]['predictions'].append(prediction_value)
            all_predictions.append(prediction_value)
            all_true_labels.append(true_label)
            prediction_count += 1

    # Aggregate predictions at the video level
    video_classifications = {}
    for video_dir, data in video_predictions.items():
        preds = data['predictions']
        true_label = data['true_labels']
        mean_prediction = np.mean(preds)
        video_classifications[video_dir] = {
            'mean_prediction': mean_prediction,
            'true_label': true_label
        }

    return video_classifications, np.array(all_true_labels), np.array(all_predictions)

if __name__ == '__main__':
    # Check for GPU, if it exists, configure it
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # Configure TensorFlow to use the GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Create the dataset
    train_df, val_df, test_df = create_datasets()

    # Load the best trained model
    best_model_loaded = load_model("results/best_cnn_model.h5")

    # Predict on the test set using the predict_videos function
    video_classifications, all_true_labels, all_predictions = predict_videos(best_model_loaded, test_df)  # Adjust the slicing as needed

    # Prepare true labels and predicted labels for confusion matrix and classification report
    y_true = []
    y_pred = []  # This will store the mean prediction values for AUC/ROC
    video_names = []

    for video_dir, result in video_classifications.items():
        video_name = os.path.basename(video_dir)  # Get the last subdirectory (video name)
        video_names.append(video_name)  # Store video names
        y_true.append(result['true_label'])

        # Store the mean prediction value for ROC analysis
        mean_prediction = result['mean_prediction']
        y_pred.append(mean_prediction)  # Store the probability

    # Create a DataFrame for the results including probability predictions
    results_df = pd.DataFrame({
        'video_name': video_names,
        'predicted_probability': y_pred,  # Store the predicted probabilities
        'actual_label': y_true
    })

    # Output the results to a CSV file
    results_df.to_csv('results/video_classification_results.csv', index=False)

    print("Results saved to 'results/video_classification_results.csv'.")