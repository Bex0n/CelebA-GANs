from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from pytorch_lightning import LightningModule

from .discriminator import Discriminator
from .generator import Generator

class WGAN(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.D = Discriminator(channels=kwargs['channels'])
        self.G = Generator(latent_dim=kwargs['latent_dim'],
                           channels=kwargs['channels'])
        self.critic_steps = kwargs['critic_steps']
        self.latent_dim = kwargs['latent_dim']

        self.automatic_optimization = False

    def sample_z(self, n) -> torch.Tensor:
        sample = torch.randn(n, self.latent_dim, device=self.device)
        return sample

    def sample_G(self, n) -> torch.Tensor:
        z = self.sample_z(n)
        return self.G(z)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        g_opt, d_opt = self.optimizers()
        x, _ = batch
        batch_size = x.size()[0]
        # Train Discriminator
        fake_batch = self.sample_G(batch_size)
        real_pred = self.D(x)
        fake_pred = self.D(fake_batch)
        d_loss = torch.mean(fake_pred) - torch.mean(real_pred)
        self.D.zero_grad(d_opt)
        self.manual_backward(d_loss)
        d_opt.step()
        with torch.no_grad():
            for param in self.D.parameters():
                param.clamp_(-0.01, 0.01)
        self.log('d_loss', d_loss, on_step=True, on_epoch=True)

        # Train Generator
        if batch_idx % self.critic_steps == 0:
            fake_batch = self.sample_G(batch_size)
            fake_pred = self.D(fake_batch)
            g_loss = -torch.mean(fake_pred)
            self.G.zero_grad(g_opt)
            self.manual_backward(g_loss)
            g_opt.step()
            self.log('g_loss', g_loss, on_step=True, on_epoch=True)
    
    def on_train_epoch_end(self):
        with torch.no_grad():
            fake_batch = self.sample_G(16)
            self.logger.experiment.add_images('generated_images', fake_batch, self.current_epoch)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=1e-5, betas=(0.5, 0.9))
        d_opt = torch.optim.Adam(self.D.parameters(), lr=1e-5, betas=(0.5, 0.9))
        return g_opt, d_opt
