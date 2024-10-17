import logging
import os
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)


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


def transform(input_path, output_path, transformations):
    """Apply a list of transformations to the video and save the result."""

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

        for transform in transformations:
            frame = transform(frame)

        out.write(frame)

    cap.release()
    out.release()


def augment_video(input_dir, output_base_dir, filename):
    """Augment the video by applying various transformations."""
    input_path = os.path.join(input_dir, filename)
    for angle in [90, 180, 270]:
        output_dir = os.path.join(output_base_dir, f"rotated_{angle}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        transform(input_path, output_path, [lambda frame: rotate_frame(frame, angle)])

    for brightness in [-50, 50]:
        output_dir = os.path.join(output_base_dir, f"brightness_{brightness}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        transform(
            input_path,
            output_path,
            [lambda frame: adjust_brightness_contrast(frame, brightness=brightness)],
        )

    for contrast in [-50, 50]:
        output_dir = os.path.join(output_base_dir, f"contrast_{contrast}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        transform(
            input_path,
            output_path,
            [lambda frame: adjust_brightness_contrast(frame, contrast=contrast)],
        )

    for angle in [90, 180, 270]:
        for brightness in [-50, 50]:
            for contrast in [-50, 50]:
                output_dir = os.path.join(
                    output_base_dir,
                    f"combo_rot_{angle}_bright_{brightness}_cont_{contrast}",
                )
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename)
                transform(
                    input_path,
                    output_path,
                    [
                        lambda frame: rotate_frame(frame, angle),
                        lambda frame: adjust_brightness_contrast(
                            frame, brightness=brightness, contrast=contrast
                        ),
                    ],
                )


def augment_videos(input_dir, output_base_dir):
    """Augment videos by applying various transformations."""
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    for filename in os.listdir(input_dir):
        if not filename.endswith((".mp4")):
            continue

        start_time = time.time()
        augment_video(input_dir, output_base_dir, filename)
        end_time = time.time()
        logging.info(f"{filename} processed in {end_time - start_time} seconds")


input_directory = "data/cropped"  # Videos of cropped faces
output_directory = "data/augmented"
augment_videos(input_directory, output_directory)
