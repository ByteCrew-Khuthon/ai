import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAutoencoder1D(nn.Module):
    def __init__(self):
        super(CNNAutoencoder1D, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # [B, 128, 258]
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),                               # [B, 128, 129]

            nn.Conv1d(128, 256, kernel_size=3, padding=1), # [B, 256, 129]
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),                               # [B, 256, 64]

            nn.Conv1d(256, 512, kernel_size=3, padding=1), # [B, 512, 64]
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(2),                               # [B, 512, 32]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),   # [B, 512, 64]
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Upsample(scale_factor=2, mode='nearest'),   # [B, 256, 128]
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Upsample(scale_factor=2, mode='nearest'),   # [B, 128, 256]
            nn.Conv1d(128, 64, kernel_size=3, padding=1),  # [B, 64, 256]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # output size: [B, 64, 256] -> pad to 258
        if decoded.shape[-1] < 258:
            decoded = F.pad(decoded, (0, 258 - decoded.shape[-1]))
        elif decoded.shape[-1] > 258:
            decoded = decoded[..., :258]

        return decoded
