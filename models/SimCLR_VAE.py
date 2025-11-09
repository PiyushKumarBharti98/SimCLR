from math import log
import torch
from torch import nn
from torch.autograd import forward_ad
import torchvision
from torchvision.models import resnet50


class ResNetEncoder(nn.Module):
    """docstring"""

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_dim = resnet.fc.in_features

    def forward(self, x):
        """docstring"""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out


class VAEHead(nn.Module):
    """docstring"""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.mu_layer = nn.Linear(input_dim, latent_dim)
        self.log_var_layer = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        """docstring"""
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        return mu, log_var


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.rand_like(std)
    return mu + eps * std


class SimpleDecoder(nn.Module):
    def __init__(
        self, latent_dim=128, start_features=2048, out_channels=3, img_size=224
    ):
        super().__init__()
        self.start_size = img_size // 16
        self.fc = nn.Linear(
            latent_dim, start_features * self.start_size * self.start_size
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                start_features, 1024, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_vae):
        x = self.fc(z_vae)
        x = x.view(x.size(0), -1, self.start_size, self.start_size)
        x_recon = self.net(x)
        return x_recon


def vae_loss_function(x_recon, x, mu, log_var):
    recon_loss = F.mse_loss(x_recon, x, reduction="sum")
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss, kld_loss
