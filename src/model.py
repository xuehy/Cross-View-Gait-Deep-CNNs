import torch as th
import torch.nn as nn


class LBNet(nn.Module):
    def __init__(self, nc=2):
        super(LBNet, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(nc, 16, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 64, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 256, kernel_size=7, stride=1)
        )
        self.mlp = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(21 * 21 * 256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(-1, 21*21*256)
        x = self.mlp(x)
        return x


if __name__ == '__main__':
    lb = LBNet()
    x = th.rand(128, 2, 126, 126)
    y = lb(x)
    print(y.shape)
