import cv2
import numpy as np

from skimage import feature
from typing import Optional


def extract_lbp_feature_from_video(
        video_path: str, 
        num_points: int = 16, 
        radius: int = 1
    ) -> Optional[np.ndarray]:
    """
    Extracts the Local Binary Patterns of the grayscaled image.

    Args:
        video_path (str): Path to the video, as a string.
        num_points (int): Number of points to be used in the LBP. Defaults to 16.
        radius (int): Radius to be taken for the LBP. Defaults to 1.

    Returns:
        Either `None` or a NumPy array representing the DFT.

    Example usage:
        video_path = "../example-output/aapnvogymq.mp3" 
        lbp_features = extract_lbp_feature_from_video(video_path)

        if lbp_features is not None:
            print(f"Extracted LBP feature shape: {lbp_features.shape}")
    """

    # Capture the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Initialize list to store LBP features for each frame
    lbp_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply LBP
        lbp = feature.local_binary_pattern(
            gray_frame, num_points, radius, method="uniform"
        )

        # Compute the histogram of LBP
        (hist, _) = np.histogram(
            lbp.ravel(), bins=np.arange(0, num_points + 3), density=True
        )

        # Append the histogram as a feature vector to the list
        lbp_features.append(hist)

    # Release video capture object
    cap.release()

    # Convert list of feature vectors to a numpy array
    lbp_features_array = np.array(lbp_features)

    return lbp_features_array
