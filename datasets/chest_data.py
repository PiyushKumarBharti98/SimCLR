import os
import csv
import argparse
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

import torchvision
from torchvision import transforms


class ChestXRayDataset(Dataset):
    """
    Chest X-Ray dataset loader for NIH ChestX-ray14.
    Accepts either:
      - a CSV with column containing image filenames
      - or a directory of images (it will list all files)
    Returns two augmented views (for SimCLR) and (optionally) labels if available.
    """

    def __init__(
        self,
        images_dir: str,
        csv_path: str = None,
        transform=None,
        return_label=False,
        image_col="Image Index",
    ):
        self.images_dir = images_dir
        self.transform = transform
        self.return_label = return_label

        if csv_path and os.path.exists(csv_path):
            # Read CSV; find image filename column
            rows = []
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
            if len(rows) == 0:
                raise RuntimeError("CSV provided but appears empty.")
            if image_col in rows[0]:
                self.imgs = [
                    os.path.join(images_dir, r[image_col]) for r in rows if r[image_col]
                ]
                # optional labels
                if self.return_label:
                    # typical NIH file has 'Finding Labels' column with 'No Finding|Effusion' strings
                    labels = [r.get("Finding Labels", "") for r in rows]
                    self.labels = labels
            else:
                # fallback: try first column
                first_key = list(rows[0].keys())[0]
                self.imgs = [
                    os.path.join(images_dir, r[first_key]) for r in rows if r[first_key]
                ]
                if self.return_label:
                    self.labels = [""] * len(self.imgs)
        else:
            # fallback: list all image files
            exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            self.imgs = [
                os.path.join(images_dir, f)
                for f in os.listdir(images_dir)
                if f.lower().endswith(exts)
            ]
            if len(self.imgs) == 0:
                raise RuntimeError(
                    "No images found in directory and no valid CSV provided."
                )
            if self.return_label:
                self.labels = [""] * len(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("L")  # single channel

        if self.transform:
            x1, x2 = self.transform(img)
        else:
            x1 = x2 = transforms.ToTensor()(img)

        if self.return_label:
            return (x1, x2), self.labels[idx]
        else:
            return (x1, x2), 0
