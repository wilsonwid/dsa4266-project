# Heavily adapted from https://pytorch.org/vision/0.11/auto_examples/plot_video_api.html
import os
import random
import torch
import torchvision
import itertools

from torchvision.datasets.folder import make_dataset
from torchvision import transforms
from typing import Optional

def find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def get_samples(root: str, extensions: tuple[str] = (".mp4",)):
    _, class_to_idx = find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)

class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(
            self, 
            root: str,
            epoch_size: Optional[int] = None,
            frame_transform: Optional[transforms.Compose] = None,
            video_transform: Optional[transforms.Compose] = None,
            clip_len: int = 16
        ):
        super().__init__()

        self.samples = get_samples(root)

        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.clip_len = clip_len

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
            metadata = vid.get_metadata()
            video_frames = []

            max_seek = metadata["video"]["duration"][0] - (self.clip_len / metadata["video"]["fps"][0])
            start = random.uniform(0., max_seek)

            for frame in itertools.islice(vid.seek(start), self.clip_len):
                if self.frame_transform:
                    video_frames.append(self.frame_transform(frame))
                else:
                    video_frames.append(frame)
                current_pts = frame["pts"]
            video = torch.stack(video_frames, 0)
            if self.video_transform:
                video = self.video_transform(video)
            output = {
                "path": path,
                "video": video,
                "target": target,
                "start": start,
                "end": current_pts
            }
            yield output
                    