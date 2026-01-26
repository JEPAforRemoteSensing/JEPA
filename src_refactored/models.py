import torch.nn as nn
import torch
from vision_transformer import vit_large, vit_base, vit_predictor
import timm
from torchvision.ops import MLP

class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=384):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=1024,
            drop_path_rate=0.1,
            img_size=120,
        )
        self.proj = MLP(1024, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)
    


class MEMPJepa(nn.Module):
    def __init__(self, in_chans1, in_chans2, patch_size, img_size):
        """Multi Encoder Multi Predictor JEPA architecture"""
        super().__init__()
        self.encoder1 = vit_base(in_chans=in_chans1, img_size=[img_size], patch_size=patch_size)
        self.encoder2 = vit_base(in_chans=in_chans2, img_size=[img_size], patch_size=patch_size)
        self.probe1 = vit_predictor(num_patches=int((img_size/patch_size)**2), embed_dim=768, use_cross_attention=True)
        self.probe2 = vit_predictor(num_patches=int((img_size/patch_size)**2), embed_dim=768, use_cross_attention=True)
    
    def forward(self, images1, images2, masks_enc=None, masks_pred=None):
        if self.training:
            z_ctx1 = self.encoder1(images1, masks_enc)
            z_ctx2 = self.encoder2(images2, masks_enc)
            z_tgt1 = self.encoder1(images1, masks_pred)
            z_tgt2 = self.encoder2(images2, masks_pred)

            z_tgt2_pred = self.probe1(z_ctx1, masks_enc, masks_pred, cross_context=z_ctx2)
            z_tgt1_pred = self.probe2(z_ctx2, masks_enc, masks_pred, cross_context=z_ctx1)

            return z_ctx1, z_ctx2, z_tgt1, z_tgt2, z_tgt1_pred, z_tgt2_pred
        
        else:
            full_img_mask = torch.arange(0, 36, device=images1.device)
            z_emb1 = self.encoder1(images1, full_img_mask)
            z_emb2 = self.encoder2(images2, full_img_mask)

            return z_emb1, z_emb2

class MMLeJEPA(nn.Module):
    def __init__(self, in_chans1, in_chans2, proj_dim=768, embed_dim=1024):
        """Multi-modal LeJEPA Architecture"""
        super().__init__()
        self.encoder1 = timm.create_model(
            'vit_base_patch8_224',
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=120,
            in_chans=in_chans1
        )
        self.proj1 = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

        self.encoder2 = timm.create_model(
            'vit_base_patch8_224',
            pretrained=False,
            num_classes=embed_dim,
            drop_path_rate=0.1,
            img_size=120,
            in_chans=in_chans2
        )
        self.proj2 = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

        self.probe = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 19))

    def forward(self, s1, s2):
        N, V = s1.shape[:2]
        emb1 = self.encoder1(s1.flatten(0, 1))
        emb2 = self.encoder2(s2.flatten(0, 1))

        yhat = self.probe(torch.cat([emb1, emb2]).detach())

        if self.training:
            return yhat, self.proj1(emb1).reshape(N, V, -1).transpose(0, 1), self.proj2(emb2).reshape(N, V, -1).transpose(0, 1)
        else:
            return yhat
