import torch
import torch.nn as nn

# class ValueCNN(nn.Module):

#     def __init__(self, obs_dim, window_size):
#         super().__init__()
#         self.window_size = window_size

#         self.conv = nn.Sequential(
#             nn.Conv1d(obs_dim, 32, 3, padding=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, 3, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#         )

#         # Retornamos ao Flatten para usar o window_size explicitamente
#         self.fc = nn.Sequential(
#             nn.Linear(64 * window_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Tanh()  # Mantemos o Tanh para ajudar com o problema do Loss alto
#         )

#     def forward(self, x):
#         # x: (batch, window, features)
#         x = x.permute(0, 2, 1)  # (batch, features, window)

#         x = self.conv(x)

#         # Flatten mantendo a dimensão temporal multiplicada pelas features
#         x = x.reshape(x.size(0), -1)

#         return self.fc(x).squeeze(-1)


class ValueCNN(nn.Module):

    def __init__(self, obs_dim, window_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(obs_dim, 32, 3, padding=1),
            nn.BatchNorm1d(32),  # Essencial para estabilizar os sinais do Go2
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * window_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Saída linear pura para o Log-Scaling
        )

    def forward(self, x):
        # x: (batch, window, obs_dim)
        x = x.permute(0, 2, 1)  # (batch, obs_dim, window)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x).squeeze(-1)
