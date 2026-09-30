import torch.nn as nn

class ONet(nn.Module):
    """输入：(N, 1)标量坐标；输出：(N, dim)位置分数。"""
    def __init__(self, dim=64, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)
    
