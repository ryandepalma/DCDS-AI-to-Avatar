import torch
import torch.nn as nn

class ArmAutoencoder(nn.Module):
    def __init__(self, input_size = 18, hidden_size = 32, latent_size = 8): # smaller values to avoid overfitting since we only have 30 videos
        super().__init__()

        # transforms sequences into vector
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.to_latent = nn.Linear(hidden_size, latent_size)

        # expands vector back into sequence
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first = True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        latent = self.to_latent(hidden[-1])

        dec_input = self.from_latent(latent)
        dec_input = dec_input.unsqueeze(1).repeat(1, x.size(1), 1) # expand
        output, _ = self.decoder(dec_input)

        return output, latent