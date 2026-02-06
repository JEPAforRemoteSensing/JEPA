import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist

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
        A = torch.randn(proj.size(-1), 1024, device=proj.device)
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

class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = 13824
        # self.backbone, self.embedding = resnet.__dict__[args.arch](
        #     zero_init_residual=True
        # )
        # self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        # x = self.projector(self.backbone(x))
        # y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

