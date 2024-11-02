import os

import cv2
from tqdm import tqdm

# Functions adapted from Junhui's cnn_dev.ipynb
# (Print statements removed to allow tqdm to work)


def load_video(video_name, folder_path="../../data/train_sample_videos"):
    video_path = os.path.join(folder_path, video_name)
    cap = cv2.VideoCapture(video_path)
    return cap


def extract_frames(cap, frame_interval=1):
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            resized_frame = cv2.resize(frame, (224, 224)) / 255.0
            frames.append(resized_frame)

        frame_count += 1

    return frames


def save_frames(frames, output_folder="./extracted_frames"):
    os.makedirs(output_folder, exist_ok=True)

    for frame_count, frame in enumerate(frames):
        output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, (frame * 255).astype("uint8"))


def extract_and_save_videos(input_dir, output_dir):
    filenames = sorted(os.listdir(input_dir))
    for filename in tqdm(filenames):
        if not filename.endswith((".mp4")):
            continue
        cap = load_video(filename, input_dir)
        frames = extract_frames(cap)
        video_output_dir = os.path.join(output_dir, filename[:-4])
        save_frames(frames, video_output_dir)


if __name__ == "__main__":
    categories = ["train", "test", "validation"]
    for cat in categories:
        extract_and_save_videos(f"data/{cat}/deepfake", f"data/{cat}_frames/deepfake")
        extract_and_save_videos(f"data/{cat}/real", f"data/{cat}_frames/real")
