import torch.nn as nn


class FashionMnistNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=5*5*16, out_features=120),
            nn.Tanh(),

            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),

            nn.Linear(in_features=84, out_features=10)
        )


    def forward(self, x):
        return self.model(x)