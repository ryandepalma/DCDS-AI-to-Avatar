"""
Laura Ozoria
Set the Long Short-Term memory autoencoder architecture.
Compresses arm movement into latent vector and reconstructs back.

More information: https://developer.nvidia.com/discover/lstm

DO NOT RUN THIS FILE.
"""

import torch
import torch.nn as nn

class ArmAutoencoder(nn.Module):
    def __init__(self, input_size = 18, hidden_size = 32, latent_size = 8): # smaller values to avoid overfitting since we only have 30 videos
        """
        Args:
            input_size: features per frame (6 joints x 3 dimensions = 18)
            hidden_size: internal LTSM size
            latent_size: size of compressed movement vector.

        NOTE: for different setting make sure that hidden_size and latent_size match
              between model.py and generate.py
        """
        super().__init__()

        # encoder --> compresses sequence into hidden_size values
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first = True)
        # reduce hidden sate to latent vector
        self.to_latent = nn.Linear(hidden_size, latent_size)

        # expands vector back into hidden for decoding
        self.from_latent = nn.Linear(latent_size, hidden_size)
        # reconstruct movement sequence
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first = True)

    def forward(self, x):
        """
        Forward through encoder and decoder.
        Args:
            x: input sequence (batch, frames, features)
        Return:
            output: reconstructed sequences
            latent: compressed movment vector
        """
        # take only final hidden data
        _, (hidden, _) = self.encoder(x)
        latent = self.to_latent(hidden[-1])

        # decode --> expand latent vector across all frames then reconstructs
        dec_input = self.from_latent(latent)
        dec_input = dec_input.unsqueeze(1).repeat(1, x.size(1), 1) # expand
        output, _ = self.decoder(dec_input)

        return output, latent