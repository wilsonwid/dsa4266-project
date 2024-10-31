import os
import random
from typing import Any, Callable, Optional

import cv2
import numpy as np
from tqdm import tqdm

from ..utils.types import FlipCode


def rotate_frame(frame: np.ndarray, angle: float) -> np.ndarray:
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
    frame: np.ndarray, brightness: float = 0.0, contrast: float = 0.0
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


def flip_frame(frame: np.ndarray, flip_code: FlipCode) -> np.ndarray:
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
    transformation: Callable[[np.ndarray, Optional[Any], Optional[Any]], np.ndarray],
) -> None:
    """
    Applies a single random transformation to the frame, then saves the result.
    Note that there is an identity transformation which keeps the frame the same.

    Args:
        input_path (str): Input path as a string.
        output_path (str): Output path as a string.
        transformation (Callable[[np.ndarray], np.ndarray])

    Returns:
        None
    """
    frame = cv2.imread(input_path)
    transformed_frame = transformation(frame)
    transformed_frame = transformed_frame.astype(np.uint8)
    cv2.imwrite(output_path, transformed_frame)


def augment_frame(
    input_dir: str | bytes | os.PathLike,
    output_dir: str | bytes | os.PathLike,
    filename: str,
) -> None:
    """
    Augments the frame by applying one random transformation and saves it.
    Note that there is an identity transformation which keeps the frame the same.

    Args:
        input_dir (str | bytes | os.PathLike): Input directory.
        output_dir (str | bytes | os.PathLike): Output directory.
        filename (str): File to be augmented.

    Returns:
        None
    """
    input_path = os.path.join(input_dir, filename)

    transformations = [
        lambda frame: frame,
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
    output_path = os.path.join(output_dir, filename)
    transform(input_path, output_path, transformation_fn)


def augment_frames(
    input_dir: str | bytes | os.PathLike, output_dir: str | bytes | os.PathLike
) -> None:
    """
    Augments frames by applying a random augmentation to each one.

    Args:
        input_dir (str | bytes | os.PathLike): Input directory containing the original frames.
        output_dir (str | bytes | os.PathLike): Output directory to save the augmented frames to.

    Returns:
        None
    """
    filenames = sorted(os.listdir(input_dir))
    random.seed(42)
    for filename in tqdm(filenames):
        if not filename.endswith((".mp4")):
            continue
        augment_frame(input_dir, output_dir, filename)


if __name__ == "__main__":
    categories = ["train", "validation"]
    for cat in categories:
        augment_frames(
            f"data/{cat}_frames/deepfake", f"data/{cat}_frames_augmented/deepfake"
        )
        augment_frames(f"data/{cat}_frames/real", f"data/{cat}_frames_augmented/real")
