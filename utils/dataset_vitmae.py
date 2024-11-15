# Heavily adapted from https://pytorch.org/vision/0.11/auto_examples/plot_video_api.html
import os
import random
import torch
import torchvision
import itertools
import numpy as np
import cv2

from torchvision.datasets.folder import make_dataset
from typing import Optional
from utils.utils import SEED
from skimage import feature
from transformers import AutoImageProcessor

torch.manual_seed(SEED)
random.seed(SEED)

def find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def get_samples(root: str, extensions: tuple[str] = (".mp4",)):
    _, class_to_idx = find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)

class VideoDatasetMAE(torch.utils.data.IterableDataset):
    def __init__(
            self, 
            root: str,
            epoch_size: Optional[int] = None,
            clip_len: int = 16,
            num_lbp_points: int = 16,
            lbp_radius: int = 1,
            image_processor: AutoImageProcessor = None
        ):
        super().__init__()

        self.samples = get_samples(root)

        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size
        self.clip_len = clip_len
        self.num_lbp_points = num_lbp_points
        self.lbp_radius = lbp_radius
        if image_processor is None:
            image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-large", use_fast=True)
        self.image_processor = image_processor

    
    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.epoch_size
        else:
            per_worker = int(self.epoch_size / worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker
            if worker_id == worker_info.num_workers - 1:
                iter_end = self.epoch_size
                
        for i in range(iter_start, iter_end, 1):
            path, target = self.samples[i]
            vid = torchvision.io.VideoReader(path, "video")

            video_frames = []
            proc_frames = []

            start = 0.

            for frame in itertools.islice(vid.seek(start), self.clip_len):
                gray_frame = cv2.cvtColor(frame["data"].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)

                dft = cv2.dft(np.float32(gray_frame), flags=cv2.DFT_COMPLEX_OUTPUT)
                dct = cv2.dct(np.float32(gray_frame))
                lbp = feature.local_binary_pattern(gray_frame, self.num_lbp_points, self.lbp_radius, method="uniform")

                dft_shift = np.fft.fftshift(dft)
                magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

                dft_feature_vector = torch.Tensor(magnitude_spectrum / np.linalg.norm(magnitude_spectrum)).unsqueeze(dim=0)
                dct_feature_vector = torch.Tensor(dct / np.linalg.norm(dct)).unsqueeze(dim=0)

                combined_frame = torch.cat([dft_feature_vector, dct_feature_vector, torch.Tensor(lbp).unsqueeze(dim=0)], dim=0).type(torch.uint8)

                video_frames.append(self.image_processor(frame["data"], return_tensors="pt")["pixel_values"].squeeze(0))
                proc_frames.append(self.image_processor(combined_frame, return_tensors="pt")["pixel_values"].squeeze(0))

                current_pts = frame["pts"]

            video = torch.stack(video_frames, 0)
            proc = torch.stack(proc_frames, 0)
            output = {
                "path": path,
                "video": video,
                "proc": proc,
                "target": target,
                "start": start,
                "end": current_pts
            }
            yield output
                    