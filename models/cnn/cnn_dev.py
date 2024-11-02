import os

import pandas as pd
# Check GPU
import tensorflow as tf
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


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

def create_tf_dataset(df, batch_size=32, shuffle=True):
    # Create a tf.data.Dataset from the DataFrame
    file_paths = df['file_path'].values
    labels = df['label'].values

    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # Map the loading and preprocessing function
    dataset = dataset.map(
        lambda file_path, label: (load_and_preprocess_image(file_path), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle and batch the dataset
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch for performance

    return dataset

def build_model(hp):
    model = Sequential()
    model.add(Conv2D(hp.Int('filters_1', min_value=32, max_value=128, step=32),
                     (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(hp.Int('filters_2', min_value=64, max_value=256, step=64),
                     (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(hp.Int('units', min_value=64, max_value=256, step=64), activation='relu'))
    model.add(Dropout(hp.Float('dropout', 0.3, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Check for GPU, if exists, configure it
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

    # Create datasets
    train_df, val_df, test_df = create_datasets()

    # To ensure no memory issues, create tf dataset
    batch_size = 64
    train_dataset = create_tf_dataset(train_df, batch_size=batch_size)
    validation_dataset = create_tf_dataset(val_df, batch_size=batch_size)

    # Set up the tuner
    csv_logger = CSVLogger('training_output.csv', append=True)

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='cnn_tuning'
    )

    # Start tuning
    tuner.search(train_dataset, validation_data=validation_dataset, epochs=10, callbacks=[csv_logger])

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # 1. Save the best model
    best_model.save("best_cnn_model.h5")  # Saves the model as an H5 file

    # 2. Print the model summary
    best_model.summary()