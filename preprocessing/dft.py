import cv2
import numpy as np


def extract_dft_feature_from_video(video_path):
    # Capture the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Initialize list to store DFT features for each frame
    dft_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply DFT to the entire frame
        dft = cv2.dft(np.float32(gray_frame), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Compute the magnitude spectrum
        magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

        # Normalize the feature vector
        feature_vector = magnitude_spectrum / np.linalg.norm(magnitude_spectrum)

        # Append the feature vector to the list
        dft_features.append(feature_vector)

    # Release video capture object
    cap.release()

    # Convert list of feature vectors to a numpy array
    dft_features_array = np.array(dft_features)

    return dft_features_array


"""
# Example usage:
video_path = "../example-output/aapnvogymq.mp4" 
dft_features = extract_dft_feature_from_video(video_path)

if dft_features is not None:
    print(f"Extracted DFT feature shape: {dft_features.shape}")
"""
