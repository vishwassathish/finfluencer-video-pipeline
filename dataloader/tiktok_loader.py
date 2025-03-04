import os

import torch
import numpy as np
from einops import rearrange
from torchvision.transforms import v2
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

class VideoFrames(Dataset):
    def __init__(self, root_dir, batch_size, transform=None):
        '''
        Args:
            root_dir (string): Directory with the video data.
        Returns:
            torch.data.Dataset object
        
        Dir structure:
        
        (root)influencer_id/
            ├── video_id/
                ├── frame_1.jpeg
                ├── frame_2.jpeg
                ├── ...
        '''
        self.transform = transform
        self.batch_size = batch_size
        self.dir = root_dir
        self.videos = {}
        self.video_list = []
        print("Accessing video frames...")
        for creator in os.listdir(root_dir):
            creator_dir = os.path.join(root_dir, creator)
            print(f"Accessing {creator}...")
            for video in os.listdir(creator_dir):
                key = f"{creator}_{video}"
                vpath = os.path.join(creator_dir, video)
                self.videos[key] = []
                
                for img in os.listdir(vpath):
                    img_path = os.path.join(vpath, img)
                    self.videos[key].append(img_path)
                
                self.video_list.append(key)
        
        
    def __len__(self):
        # total number of videos
        return len(self.video_list)
    
    def __getitem__(self, idx):
        # load the video frames
        frames = []
        video = self.video_list[idx]
        creator_id, video_id = video.split("_")
        paths = self.videos[video]
        
        for path in paths:
            frame = read_image(path).to(torch.float32) / 255
            frames.append(frame)

        if len(frames) < self.batch_size:
            # pad with zeros if there are less than 64 frames
            for _ in range(self.batch_size - len(frames)):
                frames.append(torch.zeros_like(frames[0]))
        frames = torch.stack(frames)[:self.batch_size]
        frames = self.transform(frames)
        
        return frames, creator_id, video_id
        

def video_loader(config):
    '''
    DataLoader
    NOTE: Don't use v2.ToTensor(). The behavior is ambiguous.
    Old: https://github.com/pytorch/vision/blob/4ab46e5f7585b86fb2befdc32d22d13635868c4e/torchvision/transforms/functional.py#L112-L113
    New: https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_deprecated.py
    '''

    transform = v2.Compose([
        # v2.Resize((config.resize)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = VideoFrames(config.data_dir, config.batch_size, transform)
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)
    
    return eval_loader


def inv_normalize(images):
    inverse = v2.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                std=[1/0.229, 1/0.224, 1/0.255]
            )
    images = inverse(images) * 255
    images = images.clamp(0, 255).to(torch.uint8)
    images = rearrange(images, 'c h w -> h w c')
    return images.cpu().numpy()