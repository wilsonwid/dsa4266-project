import os

import cv2
from tqdm import tqdm


def resize_video(input_folder, output_folder, size=(1080, 1080)):
    os.makedirs(output_folder, exist_ok=True)

    filenames = sorted(os.listdir(input_folder))
    for filename in tqdm(filenames):
        if not filename.endswith(".mp4"):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), size)

        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                break

            resized_frame = cv2.resize(frame, size)
            out.write(resized_frame)

        cap.release()
        out.release()


# Paths to the folders
cropped_folder = "data/cropped/"
augmented_folder = "data/augmented/"
output_cropped_folder = "data/cropped_resized/"
output_augmented_folder = "data/augmented_resized/"

# Resize videos in both folders
resize_video(cropped_folder, output_cropped_folder)
resize_video(augmented_folder, output_augmented_folder)
