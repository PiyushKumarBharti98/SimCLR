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

        # define transform steps appropriate for X-rays
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.1)),
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomRotation(10),  # small rotations
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=0.0),
                transforms.RandomHorizontalFlip(p=self.hflip_prob),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )

        # convert to tensor and normalize (ImageNet mean/std if repeating channels)
        self.to_tensor = transforms.ToTensor()
        # If using 3-channel ImageNet-pretrained ViT, use ImageNet norm; else normalize single channel to [0,1] mean/std
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


class ViTBackbone(nn.Module):
    """
    Wrapper: ViT encoder (returns representation h).
    Supports timm or torchvision ViT.
    """

    def __init__(
        self,
        model_name="vit_base_patch16_224",
        pretrained=True,
        use_timm=True,
        in_chans=3,
    ):
        super().__init__()
        self.use_timm = use_timm and has_timm
        self.in_chans = in_chans
        if self.use_timm:
            # timm: e.g., 'vit_base_patch16_224'
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
                in_chans=in_chans,
            )
            feat_dim = self.model.num_features
        else:
            if model_name.startswith("vit_b"):
                base = torchvision.models.vit_b_16(pretrained=pretrained)
            else:
                base = torchvision.models.resnet50(pretrained=pretrained)
            if hasattr(base, "heads"):
                feat_dim = base.heads.head.in_features
                base.heads = nn.Identity()
            else:
                feat_dim = base.fc.in_features
                base.fc = nn.Identity()
            self.model = base

        self.feat_dim = feat_dim

    def forward(self, x):
        h = self.model(x)
        return h
