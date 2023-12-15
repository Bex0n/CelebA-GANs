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
        self.gp_lambda = kwargs['gp_lambda']
        self.d_lr = kwargs['d_lr']
        self.g_lr = kwargs['g_lr']

        self.automatic_optimization = False

    def sample_z(self, n) -> torch.Tensor:
        sample = torch.randn(n, self.latent_dim, device=self.device)
        return sample

    def sample_G(self, n) -> torch.Tensor:
        z = self.sample_z(n)
        return self.G(z)

    def compute_gp(self, batch: torch.Tensor, fake_batch: torch.Tensor) -> float:
        batch_size = batch.size(0)
        eps = torch.rand(batch_size, 1, 1, 1).to(self.device)
        eps = eps.expand_as(batch)

        interpolation = eps * batch + (1 - eps) * fake_batch

        interpolation_output = self.D(interpolation)
        grad_outputs = torch.ones_like(interpolation_output)

        # Compute Gradients
        gradients = torch.autograd.grad(
            outputs=interpolation_output,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        return torch.mean((grad_norm - 1) ** 2)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        g_opt, d_opt = self.optimizers()
        self.D.zero_grad(d_opt)
        x, _ = batch
        batch_size = x.size()[0]
        # Train Discriminator
        fake_batch = self.sample_G(batch_size)
        real_pred = self.D(x)
        fake_pred = self.D(fake_batch)
        gp_loss = self.compute_gp(x, fake_batch)
        em_loss = torch.mean(fake_pred) - torch.mean(real_pred)
        d_loss = em_loss + self.gp_lambda * gp_loss
        self.manual_backward(d_loss)
        d_opt.step()
        self.log('real_d_loss', -torch.mean(real_pred), on_step=False, on_epoch=True)
        self.log('fake_d_loss', torch.mean(fake_pred), on_step=False, on_epoch=True)
        self.log('gp_loss', gp_loss, on_step=False, on_epoch=True)
        self.log('d_loss', d_loss, on_step=False, on_epoch=True)

        # Train Generator
        if batch_idx % self.critic_steps == 0:
            fake_batch = self.sample_G(batch_size)
            fake_pred = self.D(fake_batch)
            g_loss = -torch.mean(fake_pred)
            self.G.zero_grad(g_opt)
            self.manual_backward(g_loss)
            g_opt.step()
            self.log('g_loss', g_loss, on_epoch=True)

    def on_validation_epoch_end(self):
        fake_batch = self.sample_G(32)
        self.logger.experiment.add_images('generated_images', fake_batch, self.current_epoch)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=self.g_lr, betas=(0.5, 0.9))
        d_opt = torch.optim.Adam(self.D.parameters(), lr=self.d_lr, betas=(0.5, 0.9))
        return g_opt, d_opt
