import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.channels = kwargs['channels']

        self.hidden_dims = [self.channels, 16, 32, 64, 128, 256]
        modules = []

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.hidden_dims[i],
                              out_channels=self.hidden_dims[i + 1],
                              kernel_size=4,
                              padding=1,
                              stride=2),
                    nn.LeakyReLU()
                )
            )
        self.conv = nn.Sequential(*modules)
        self.fc = nn.Sequential(
            nn.Linear(in_features=256 * 2 * 2, out_features=100),
            nn.Sigmoid(),
            nn.Linear(in_features=100, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
