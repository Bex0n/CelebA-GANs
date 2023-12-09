import torch
from pytorch_lightning import LightningModule

from .discriminator import Discriminator
from .generator import Generator

class WGAN(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.G = Discriminator(channels=kwargs['channels'])
        self.D = Generator(latent_dim=kwargs['latent_dim'],
                           channels=kwargs['channels'])
        self.critics_steps = kwargs['critics_steps']

        self.automatic_optimization = False

    def forward(self, x):
        pass

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        for i in range(self.critics_steps):
            batch_size = batch.shape[0]
            fake_noise = torch.randn(batch_size, self.latent_dim,
                                     device=self.device)
            fake_batch = self.G(fake_noise)
            real_pred = self.D(batch)
            fake_pred = self.D(fake_batch)
            # Train Discriminator
            with torch.no_grad():
                for param in self.D.parameters():
                    param.clamp_(-0.01, 0.01)
            # Train Generator
        pass

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=1e-5)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=1e-5)
        return g_opt, d_opt
