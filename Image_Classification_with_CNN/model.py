import torch.nn as nn
from pdb import set_trace as bp
class FashionMNISTNet(nn.Module):

    """ Simple network"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(0.4),
            nn.Conv2d(64, 128, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(128, 256, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),

            # nn.ELU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(25 * 25 * 256, 1024),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.features(x)
        # bp()
        x = x.view(-1, 256*25*25)
        x = self.classifier(x)
        # bp()
        return x
