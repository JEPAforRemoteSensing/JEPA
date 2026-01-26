import torch.nn as nn
import torch
from vision_transformer import vit_large, vit_base, vit_predictor

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

