import torch.nn as nn
import torch
import torch.nn.functional as F

class MEMPLoss(nn.Module):
    def __init__(self):
        super().__init__()
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

    def forward(self, z_ctx1, z_ctx2, z_tgt1, z_tgt2, z_tgt1_pred, z_tgt2_pred):
        proj = torch.cat([z_ctx1.unsqueeze(0), z_ctx2.unsqueeze(0), z_tgt1.unsqueeze(0), z_tgt2.unsqueeze(0)], dim=0)
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)


        sigreg_loss = statistic.mean()

        probe1_loss = F.mse_loss(z_tgt1_pred, z_tgt1.detach()) + F.mse_loss(z_tgt1_pred, z_tgt2.detach())
        probe2_loss = F.mse_loss(z_tgt2_pred, z_tgt2.detach()) + F.mse_loss(z_tgt2_pred, z_tgt1.detach())

        return sigreg_loss, probe1_loss, probe2_loss

class LeJEPALoss(nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj, yhat, y_rep):
        A = torch.randn(proj.size(-1), 512, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        
        sigreg_loss = statistic.mean()
        inv_loss = (proj.mean(0) - proj).square().mean()
        probe_loss = F.binary_cross_entropy_with_logits(yhat, y_rep)

        return sigreg_loss, inv_loss, probe_loss

