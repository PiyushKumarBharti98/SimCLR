import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import random


class SimCLRXRayTransform:
    """
    Generate two augmented views for a single-channel (grayscale) chest x-ray image.
    """

    def __init__(self, image_size=224, to_3ch=True, hflip_prob=0.0):
        self.image_size = image_size
        self.to_3ch = to_3ch
        self.hflip_prob = hflip_prob

        self.base_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.1)),
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=0.0),
                transforms.RandomHorizontalFlip(p=self.hflip_prob),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )

        # convert to tensor and normalize (ImageNet mean/std if repeating channels)
        self.to_tensor = transforms.ToTensor()
        self.normalize_3ch = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        self.normalize_1ch = transforms.Normalize(mean=(0.5,), std=(0.25,))

        # gaussian blur/noise as separate small transforms
        self.gaussian_blur = transforms.GaussianBlur(
            kernel_size=(5, 5), sigma=(0.1, 2.0)
        )
        self.noise = lambda x: x + 0.01 * torch.randn_like(x)

    def __call__(self, img: Image.Image):
        v1 = self.base_transform(img)
        v2 = self.base_transform(img)

        t1 = self.to_tensor(v1)  # single channel tensor [1,H,W]
        t2 = self.to_tensor(v2)

        if random.random() < 0.2:
            t1 = self.gaussian_blur(t1)
        if random.random() < 0.2:
            t2 = self.gaussian_blur(t2)

        if random.random() < 0.1:
            t1 = self.noise(t1)
        if random.random() < 0.1:
            t2 = self.noise(t2)

        if self.to_3ch:
            t1 = t1.repeat(3, 1, 1)
            t2 = t2.repeat(3, 1, 1)
            t1 = self.normalize_3ch(t1)
            t2 = self.normalize_3ch(t2)
        else:
            t1 = self.normalize_1ch(t1)
            t2 = self.normalize_1ch(t2)

        return t1, t2


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResNetEncoder(nn.Module):
    """docstring"""

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        resnet = resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_dim = resnet.fc.in_features

    def forward(self, x):
        """docstring"""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out


class NTXentLoss(nn.Module):
    def __init__(self, batch_size: int, temperature: float = 0.1, device="cuda"):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.register_buffer("mask", self._create_mask())

    def _create_mask(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask.fill_diagonal_(False)
        for i in range(self.batch_size):
            mask[i, i + self.batch_size] = False
            mask[i + self.batch_size, i] = False
        return mask

    def forward(self, z1, z2):
        assert z1.shape == z2.shape
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # 2B x D
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature  # 2B x 2B

        positives = torch.cat(
            [torch.diag(sim, B), torch.diag(sim, -B)], dim=0
        ).unsqueeze(
            1
        )  # 2B x 1
        negatives = sim[self.mask].view(2 * B, -1)  # 2B x (2B-2)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * B, dtype=torch.long).to(self.device)
        loss = self.criterion(logits, labels)
        return loss
