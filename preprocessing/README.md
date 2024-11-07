## Preprocessing

### Balanced Dataset Creation

To address class imbalance in the DFDC dataset, we performed undersampling on the deepfake videos, retaining all real videos and randomly selecting a subset of deepfake videos to match the count of real videos.

### Face Extraction Using YOLO

Since only faces were deepfaked, we used a YOLOv11 model, fine-tuned for face detection, to crop faces in the videos to 224x224 pixels, with a margin around the face. Videos with multiple faces were excluded to simplify analysis, resulting in a balanced dataset of 2,750 real and 2,750 deepfake videos.

### Train-Test Split

We divided the dataset into train, test, and validation sets in a 0.65:0.2:0.15 ratio. Deepfake videos from the same real video were not intentionally grouped or separated across the splits.

### Frame-based Model Preparation

For frame-based models, additional preprocessing included extracting frames from each video (300 frames per video) and applying data augmentation on the train and validation sets. Augmentation included random transformations like rotation, brightness, and contrast adjustments, while the test set was left unaltered.

### Extra Features

We explored incorporating additional feature inputs for the machine learning models, including the Discrete Fourier Transform (DFT), Discrete Cosine Transform (DCT), and Local Binary Patterns (LBP).
