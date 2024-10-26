import os
import random
import cv2
import numpy as np

from tqdm import tqdm
from ..utils.types import FlipCode
from typing import Any, Callable, Optional

def rotate_frame(
        frame: np.ndarray, 
        angle: float
    ) -> np.ndarray:
    """
    Rotates the given `frame` by the angle `angle`.

    Args:
        frame (np.ndarray): Original frame to be transformed.
        angle (float): Angle to rotate the frame by.

    Returns:
        Rotated array of type `np.ndarray`.
    """
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, matrix, (width, height))
    return rotated


def adjust_brightness_contrast(
        frame: np.ndarray, 
        brightness: float = 0.0, 
        contrast: float = 0.0
    ) -> np.ndarray:
    """
    Adjusts the brightness and contrast of a frame.

    Args:
        frame (np.ndarray): Original frame to be transformed.
        brightness (float): Amount to increase the brightness by.
        contrast (float): Amount fo increase the contrast by.

    Returns:
        Modified array of type `np.ndarray`.
    """
    frame = np.clip(frame * (1 + contrast / 100.0) + brightness, 0, 255).astype(
        np.uint8
    )
    return frame


def flip_frame(
        frame: np.ndarray,
        flip_code: FlipCode
    ) -> np.ndarray:
    """
    Flips the frame.

    Args:
        frame (np.ndarray): Original frame to be transformed.
        flip_code (FlipCode): Flip code; 1 for vertical flip and 0 for horizontal flip.
    
    Returns:
        Flipped frame of type `np.ndarray`.
    """
    return cv2.flip(frame, flip_code)


def transform(
        input_path: str,
        output_path: str,
        transformation: Callable[[np.ndarray, Optional[Any], Optional[Any]], np.ndarray]
    ) -> None:
    """
    Applies a single random transformation to the video, then saves the result.

    Args:
        input_path (str): Input path as a string.
        output_path (str): Output path as a string.
        transformation (Callable[[np.ndarray], np.ndarray])

    Returns:
        None
    """
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


def augment_video(
        input_dir: str | bytes | os.PathLike,
        output_dir: str | bytes | os.PathLike, 
        filename: str
    ) -> None:
    """
    Augments the video by applying one random transformation and saves it.

    Args:
        input_dir (str | bytes | os.PathLike): Input directory.
        output_dir (str | bytes | os.PathLike): Output directory.
        filename (str): File to be augmented.

    Returns:
        None
    """
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


def augment_videos(
        input_dir: str | bytes | os.PathLike, 
        output_dir: str | bytes | os.PathLike
    ) -> None:
    """
    Augments videos by applying a random augmentation to each one.

    Args:
        input_dir (str | bytes | os.PathLike): Input directory containing the original videos.
        output_dir (str | bytes | os.PathLike): Output directory to save the augmented videos to.
    
    Returns:
        None
    """
    filenames = sorted(os.listdir(input_dir))
    random.seed(42)
    for filename in tqdm(filenames):
        if not filename.endswith((".mp4")):
            continue
        augment_video(input_dir, output_dir, filename)


if __name__ == "__main__":
    augment_videos("data/train/deepfake", "data/train/deepfake")
    augment_videos("data/train/real", "data/train/real")
