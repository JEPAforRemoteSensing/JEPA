import torch.nn as nn
import torch

class MEMPLoss(nn.Module):
    def __init__(self, lamb, gamma):
        super().__init__()
        self.lamb = lamb
        self.gamma = gamma

        # SIGReg Loss
        knots = 17
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

        self.__name__ = 'MEMPLoss'

    def forward(self):
        pass

