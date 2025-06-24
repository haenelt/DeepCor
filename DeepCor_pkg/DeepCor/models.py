#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .utils import compute_padding


__all__ = ["DeepCor_model"]


class DeepCor_model(nn.Module):
    """This model initializes two encoders (signal and noise respectively) and one
    decoder."""

    def __init__(
        self,
        in_channels: int,
        in_dim: int,
        latent_dim: int,
        hidden_dims: list = [64, 128, 256, 256],
    ) -> None:
        super().__init__()

        self.in_channels = in_channels  # will be set to 1
        self.in_dim = in_dim  # the length of time courses of fMRI
        self.hidden_dims = hidden_dims  # number of neurons for each hidden layers
        self.latent_dim = latent_dim  # the bottleneck dimension

        self.pad, self.final_size, self.pad_out = compute_padding(self.in_dim)

        # Build Encoder (four layers of 1D CNN)
        modules_z = []
        for i, _ in enumerate(hidden_dims):
            h_dim = hidden_dims[i]
            modules_z.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=int(self.pad[-i - 1]),
                    ),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder_z = nn.Sequential(*modules_z)
        self.fc_mu_z = nn.Linear(hidden_dims[-1] * int(self.final_size), latent_dim)
        self.fc_var_z = nn.Linear(hidden_dims[-1] * int(self.final_size), latent_dim)

        modules_s = []
        in_channels = self.in_channels
        for i, _ in enumerate(hidden_dims):
            h_dim = hidden_dims[i]
            modules_s.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=int(self.pad[-i - 1]),
                    ),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder_s = nn.Sequential(*modules_s)
        self.fc_mu_s = nn.Linear(hidden_dims[-1] * int(self.final_size), latent_dim)
        self.fc_var_s = nn.Linear(hidden_dims[-1] * int(self.final_size), latent_dim)

        # Build Decoder (four layers of 1D Reverse CNN)
        self.decoder_input = nn.Linear(
            2 * latent_dim, hidden_dims[-1] * int(self.final_size)
        )
        hidden_dims.reverse()

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=int(self.pad_out[-4 + i]),
                        output_padding=int(self.pad_out[-4 + i]),
                    ),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=int(self.pad_out[-1]),
                output_padding=int(self.pad_out[-1]),
            ),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dims[-1], out_channels=1, kernel_size=3, padding=1),
        )

    def encode_z(self, input_: Tensor) -> list[Tensor]:
        """Encodes the input by passing through the signal encoder network and returns
        the latent codes."""
        result = self.encoder_z(input_)
        result = torch.flatten(result, start_dim=1)

        # split the result into mu and var components of the latent Gaussian
        # distribution
        mu = self.fc_mu_z(result)
        logvar = self.fc_var_z(result)

        return [mu, logvar]

    def encode_s(self, input_: Tensor) -> list[Tensor]:
        """Encodes the input by passing through the noise encoder network and returns
        the latent codes."""
        result = self.encoder_s(input_)
        result = torch.flatten(result, start_dim=1)

        # split the result into mu and var components of the latent Gaussian
        # distribution
        mu = self.fc_mu_s(result)
        logvar = self.fc_var_s(result)
        return [mu, logvar]

    def decode(self, z: Tensor) -> Tensor:
        """Maps the given latent codes onto the original space."""
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], int(self.final_size))
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward_tg(self, input_: Tensor) -> list[Tensor]:
        """Forwards the input that encodes by both signal and noise encoder,
        reparameterize, and decodes by the decoder. Used during the training for voxels
        from ROI."""
        tg_mu_z, tg_logvar_z = self.encode_z(input_)
        tg_mu_s, tg_logvar_s = self.encode_s(input_)
        tg_z = self.reparameterize(tg_mu_z, tg_logvar_z)
        tg_s = self.reparameterize(tg_mu_s, tg_logvar_s)
        output = self.decode(torch.cat((tg_z, tg_s), 1))
        return [output, input_, tg_mu_z, tg_logvar_z, tg_mu_s, tg_logvar_s, tg_z, tg_s]

    def forward_bg(self, input_: Tensor) -> list[Tensor]:
        """Forwards the input that encodes by only the noise encoder, reparameterize,
        and decodes by the decoder. Used during the training for voxels from RONI."""
        bg_mu_s, bg_logvar_s = self.encode_s(input_)
        bg_s = self.reparameterize(bg_mu_s, bg_logvar_s)
        zeros = torch.zeros_like(bg_s)
        output = self.decode(torch.cat((zeros, bg_s), 1))
        return [output, input_, bg_mu_s, bg_logvar_s]

    def forward_fg(self, input_: Tensor) -> list[Tensor]:
        """Forwards the input that encodes by only the signal encoder, reparameterize,
        and decodes by the decoder. Used during denoising the voxels from ROI."""
        fg_mu_z, fg_logvar_z = self.encode_z(input_)
        tg_z = self.reparameterize(fg_mu_z, fg_logvar_z)
        zeros = torch.zeros_like(tg_z)
        output = self.decode(torch.cat((tg_z, zeros), 1))
        return [output, input_, fg_mu_z, fg_logvar_z]

    def loss_function(
        self,
        *args,
    ) -> dict:
        """Computes the VAE loss function. KL(N(\mu, \sigma), N(0, 1)) =
        \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}."""
        beta = 0.00001

        recons_tg = args[0]
        input_tg = args[1]
        tg_mu_z = args[2]
        tg_logvar_z = args[3]
        tg_mu_s = args[4]
        tg_logvar_s = args[5]
        recons_bg = args[8]
        input_bg = args[9]
        bg_mu_s = args[10]
        bg_logvar_s = args[11]

        recons_loss = F.mse_loss(recons_tg, input_tg)
        recons_loss += F.mse_loss(recons_bg, input_bg)

        kld_loss = 1 + tg_logvar_z - tg_mu_z**2 - tg_logvar_z.exp()
        kld_loss += 1 + tg_logvar_s - tg_mu_s**2 - tg_logvar_s.exp()
        kld_loss += 1 + bg_logvar_s - bg_mu_s**2 - bg_logvar_s.exp()
        kld_loss = torch.mean(-0.5 * torch.sum(kld_loss, dim=1), dim=0)

        loss = torch.mean(recons_loss + beta * kld_loss)
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": kld_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: int) -> Tensor:
        """Samples from the latent space and return the corresponding image space
        map."""
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor) -> Tensor:
        """Given a voxel from ROI, returns the denoised voxel."""
        return self.forward_fg(x)[0]
