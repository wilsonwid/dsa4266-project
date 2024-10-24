import os
import random

import cv2
import numpy as np
from tqdm import tqdm


def rotate_frame(frame, angle):
    """Rotate the given frame by the specified angle."""
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, matrix, (width, height))
    return rotated


def adjust_brightness_contrast(frame, brightness=0, contrast=0):
    """Adjust the brightness and contrast of a frame."""
    frame = np.clip(frame * (1 + contrast / 100.0) + brightness, 0, 255).astype(
        np.uint8
    )
    return frame


def flip_frame(frame, flip_code):
    """Flip the frame. flip_code 1 for horizontal, 0 for vertical."""
    return cv2.flip(frame, flip_code)


def transform(input_path, output_path, transformation):
    """Apply a single transformation to the video and save the result."""
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        has_frame, frame = cap.read()
        if not has_frame:
            break

        frame = transformation(frame)
        out.write(frame)

    cap.release()
    out.release()


def augment_video(input_dir, output_dir, filename):
    """Augment the video by applying one random transformation."""
    input_path = os.path.join(input_dir, filename)

    transformations = [
        lambda frame: rotate_frame(frame, 90),
        lambda frame: rotate_frame(frame, 180),
        lambda frame: rotate_frame(frame, 270),
        lambda frame: adjust_brightness_contrast(frame, brightness=-25),
        lambda frame: adjust_brightness_contrast(frame, brightness=25),
        lambda frame: adjust_brightness_contrast(frame, contrast=-25),
        lambda frame: adjust_brightness_contrast(frame, contrast=25),
        lambda frame: flip_frame(frame, 1),
        lambda frame: flip_frame(frame, 0),
    ]

    transformation_fn = random.choice(transformations)
    output_path = os.path.join(output_dir, filename[:-4] + "_augmented.mp4")
    transform(input_path, output_path, transformation_fn)


def augment_videos(input_dir, output_dir):
    """Augment videos by applying a random transformation to each one."""
    filenames = sorted(os.listdir(input_dir))
    random.seed(42)
    for filename in tqdm(filenames):
        if not filename.endswith((".mp4")):
            continue
        augment_video(input_dir, output_dir, filename)


augment_videos("data/train/deepfake", "data/train/deepfake")
augment_videos("data/train/real", "data/train/real")
