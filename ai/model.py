import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Encoder: [B, 64, 44] → [B, 512, 6]
        self.encoder = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),   # [B, 128, 22]
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),  # [B, 256, 11]
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),  # [B, 512, 6]
            nn.ReLU(),
        )

        # Decoder: [B, 512, 6] → [B, 64, 44]
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 12]
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 24]
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),   # [B, 64, 48]
            nn.Sigmoid()  # [0, 1] 범위에 맞춰서 출력
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # 출력 시 원래 크기 [B, 64, 44]로 자르기
        decoded = decoded[:, :, :44]
        return decoded