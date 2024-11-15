import cv2
import numpy as np

from typing import Optional


def extract_dct_feature_from_video(video_path: str) -> Optional[np.ndarray]:
    """
    Extracts the Discrete Cosine Transform (DCT) from the grayscale of the image.

    Args:
        video_path (str): Path to the video as a string.

    Returns:
        Either `None` or a NumPy array representing the DCT.

    Example Usage:
        video_path = "../example-output/aapnvogymq.mp4" 
        dct_features = extract_dct_feature_from_video(video_path)
    """
    # Capture the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Initialize list to store DCT features for each frame
    dct_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply DCT to the entire frame
        dct = cv2.dct(np.float32(gray_frame))

        # Normalize the feature vector
        feature_vector = dct / np.linalg.norm(dct)

        # Append the feature vector to the list
        dct_features.append(feature_vector)

    # Release video capture object
    cap.release()

    # Convert list of feature vectors to a NumPy array
    dct_features_array = np.array(dct_features)

    return dct_features_array
