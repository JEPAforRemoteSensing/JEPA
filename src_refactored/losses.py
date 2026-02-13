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

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        if self.clip is not None and self.clip > 0:
            probs = torch.clamp(probs, min=self.clip, max=1 - self.clip)

        pos_loss = targets * torch.log(probs + self.eps)
        neg_loss = (1 - targets) * torch.log(1 - probs + self.eps)

        pos_loss = pos_loss * (1 - probs).pow(self.gamma_pos)
        neg_loss = neg_loss * probs.pow(self.gamma_neg)

        loss = -(pos_loss + neg_loss)
        return loss.mean()


class LeJEPALoss(nn.Module):
    def __init__(self, num_views, knots=17, t_max=5, num_slices=256, use_asl=False, gamma_neg=4, gamma_pos=0, clip=0.05):
        super().__init__()
        self.num_views = num_views
        self.num_slices = num_slices
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        
        self.use_asl = use_asl
        if use_asl:
            self.asl_loss = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)

    def forward(self, proj, yhat, y_rep):
        """Args:
            proj: (2V, N, K) concatenated projected embeddings from both encoders.
                  SIGReg is applied per-view (mean over N per view), so each
                  modality's views are regularized independently. The invariance
                  loss computes a cross-modal center (mean over all 2V views),
                  pulling S1 and S2 embeddings of the same sample together.
            yhat: (2*V*N, num_classes) probe predictions
            y_rep: (2*V*N, num_classes) repeated labels
        """
        # SIGReg: per-view, then averaged over all 2V views
        A = torch.randn(proj.size(-1), self.num_slices, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True))
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (2V, N, num_slices, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()  # mean over N
        statistic = (err @ self.weights) * proj.size(-2)  # (2V, num_slices)
        sigreg_loss = statistic.mean()

        # Invariance/prediction loss:
        # Center = mean over ALL 2V views (cross-modal center) per sample
        # This is what pulls S1 and S2 representations together
        inv_loss = (proj[:self.num_views] - proj[:self.num_views].mean(0)).square().mean() + (proj[self.num_views:] - proj[self.num_views:].mean(0)).square().mean()
        center_loss = (proj[:self.num_views].mean(0) - proj[self.num_views:].mean(0)).square().mean()

        if self.use_asl:
            probe_loss = self.asl_loss(yhat, y_rep.float())
        else:
            probe_loss = F.binary_cross_entropy_with_logits(yhat, y_rep.float())

        return sigreg_loss, inv_loss, probe_loss, center_loss

