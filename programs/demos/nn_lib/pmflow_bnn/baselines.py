import torch.nn as nn

class CNNBaseline(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*7*7, 256), nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.net(x)

class GRUBaseline(nn.Module):
    def __init__(self, hidden=128, n_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size=28, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        seq = x.squeeze(1)  # (B, 28, 28)
        out, _ = self.gru(seq)
        return self.fc(out[:, -1, :])
