import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.latent_dim = kwargs['latent_dim']
        self.channels = kwargs['channels']

        self.hidden_dims = [256, 128, 64, 32]
        modules = []

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dims[0] * 4 * 4),
            nn.ReLU()
        )

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1],
                               self.channels,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.Sigmoid()
        ))
        self.conv = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(-1, self.hidden_dims[0], 4, 4)
        x = self.conv(x)
        return x
