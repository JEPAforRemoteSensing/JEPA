import torch.nn as nn
import torch
from vision_transformer import vit_base, vit_predictor

class MEMPJepa(nn.Module):
    def __init__(self, in_chans1, in_chans2):
        """Multi Encoder Multi Predictor JEPA architecture"""
        super().__init__()
        self.encoder1 = vit_base(in_chans=in_chans1)
        self.encoder2 = vit_base(in_chans2=in_chans2)
        self.probe1 = vit_predictor(num_patches=16)
        self.probe2 = vit_predictor(num_patches=16)


