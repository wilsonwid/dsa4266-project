## Preprocessing

### File Descriptions

#### 1. `augment.py`

- **Purpose**: This script provides functions for augmenting video frames, including rotation and flipping.
- **Implementation**:
  - **Rotation**: Rotates frames by a specified angle using `cv2`.
  - **Flipping**: Supports horizontal and vertical flipping, with `FlipCode` for direction control.
  - Utilizes `tqdm` for progress tracking.

#### 2. `create_balanced_dataset.ipynb`

- **Purpose**: Creates a balanced dataset by sampling an equal number of instances from each class.
- **Implementation**:
  - Uses `pandas` to load data and random sampling for balancing classes.
  - Iterates through the dataset to segregate samples and create a balanced set for training.

#### 3. `dct.py`

- **Purpose**: Extracts Discrete Cosine Transform (DCT) features from video frames.
- **Implementation**:
  - Converts video frames to grayscale and applies DCT using `numpy`.
  - Outputs the DCT features as a numerical representation of each frame for further analysis.

#### 4. `dft.py`

- **Purpose**: Extracts Discrete Fourier Transform (DFT) features from video frames.
- **Implementation**:
  - Similar to `dct.py`, it processes video frames in grayscale and computes DFT.
  - The extracted features help analyze frequency components, useful for detecting specific patterns.

#### 5. `frames.py`

- **Purpose**: Manages frame extraction from videos.
- **Implementation**:
  - Loads videos and iterates through frames using OpenCV.
  - Supports saving frames for batch processing and analysis, with a progress bar for user convenience.

#### 6. `label.py`

- **Purpose**: Handles labeling and directory organization of deepfake and real videos.
- **Implementation**:
  - Reads metadata to distinguish between real and fake samples.
  - Organizes video samples into respective directories for training and testing.

#### 7. `lbp.py`

- **Purpose**: Extracts Local Binary Patterns (LBP) features, which are useful for texture analysis.
- **Implementation**:
  - Processes grayscale frames and calculates LBP features with adjustable parameters (e.g., number of points, radius).
  - Uses `skimage` to compute and store the LBP features.

#### 8. `split.py`

- **Purpose**: Splits the dataset into training, validation, and test sets.
- **Implementation**:
  - Moves files into respective directories based on specified ratios for training, validation, and testing.
  - Supports customizable ratios for flexible dataset splitting.

#### 9. `yolo.py`

- **Purpose**: Extracts faces from videos using the YOLO (You Only Look Once) model for object detection.
- **Implementation**:
  - Uses a pretrained YOLO model to detect and crop faces from video frames.
  - Processes videos in batches, saving cropped faces to a specified directory.

---

### Summary

The full data flow pipeline is outlined:
<img src="./Data%20flow%20pipeline.png" alt="Data Flow Pipeline" width="50%">

#### Balanced Dataset Creation

To address class imbalance in the DFDC dataset, we performed undersampling on the deepfake videos, retaining all real videos and randomly selecting a subset of deepfake videos to match the count of real videos.

#### Face Extraction Using YOLO

Since only faces were deepfaked, we used a YOLOv11 model, fine-tuned for face detection, to crop faces in the videos to 224x224 pixels, with a margin around the face. Videos with multiple faces were excluded to simplify analysis, resulting in a balanced dataset of 2,750 real and 2,750 deepfake videos.

#### Train-Test Split

We divided the dataset into train, test, and validation sets in a 0.65:0.2:0.15 ratio. Deepfake videos from the same real video were not intentionally grouped or separated across the splits.

#### Frame-based Model Preparation

For frame-based models, additional preprocessing included extracting frames from each video (300 frames per video) and applying data augmentation on the train and validation sets. Augmentation included random transformations like rotation, brightness, and contrast adjustments, while the test set was left unaltered.

#### Extra Features

We explored incorporating additional feature inputs for the machine learning models, including the Discrete Fourier Transform (DFT), Discrete Cosine Transform (DCT), and Local Binary Patterns (LBP).