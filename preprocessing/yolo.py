import os

import cv2
from ultralytics import YOLO

model = YOLO("yolov11n-face.pt")
input_path = "../data/train_sample_videos/"
SCALE = 1.5


def extract_face_from_video(folder_path, video_name, output_path="../cropped/"):
    # Open the input video
    cap = cv2.VideoCapture(folder_path + video_name)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter to save the output video with faces
    output_size = (224, 224)  # square output
    output_file = output_path + video_name
    out = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    double_face_detected = False

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO face detection on the frame
        results = model(frame)

        # Extract the first face detected
        if len(results[0].boxes) == 1:
            # Get bounding box of the first face detected
            boxes = results[0].boxes
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Take 1.5 * area surrounding the face
            width = x2 - x1
            height = y2 - y1
            x_center = x1 + width / 2
            y_center = y1 + height / 2

            side_length = max(width, height) * SCALE

            x1 = max(int(x_center - side_length / 2), 0)
            y1 = max(int(y_center - side_length / 2), 0)
            x2 = min(int(x_center + side_length / 2), frame_width)
            y2 = min(int(y_center + side_length / 2), frame_height)

            # Crop the face from the frame
            face_frame = frame[y1:y2, x1:x2]

            # Resize the cropped face to the original frame size for consistency
            face_frame_resized = cv2.resize(face_frame, output_size)

            # Write the frame with the face to the output video
            out.write(face_frame_resized)
        else:
            double_face_detected = True
            break

    cap.release()
    out.release()
    if double_face_detected and os.path.exists(output_file):
        os.remove(output_file)


def process_videos_from_folder(folder_path, output_path="../cropped/"):
    videos = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(".mp4")
    ]
    os.makedirs(output_path, exist_ok=True)
    for video_name in videos:
        extract_face_from_video(folder_path, video_name, output_path)


process_videos_from_folder(input_path)
