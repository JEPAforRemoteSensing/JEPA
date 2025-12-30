# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder module for I-JEPA.

In I-JEPA, the encoder is a Vision Transformer (ViT) that processes masked context blocks.
The same architecture is used for both:
- Context Encoder: processes visible context patches (trained with gradients)
- Target Encoder: processes full image to produce targets (updated via EMA)
"""

from vision_transformer import (
    VisionTransformer,
    vit_tiny,
    vit_small,
    vit_base,
    vit_large,
    vit_huge,
    vit_giant,
    VIT_EMBED_DIMS,
)


def build_encoder(
    model_name='vit_base',
    img_size=224,
    patch_size=16,
    in_chans=3,
    drop_path_rate=0.0,
    **kwargs
):
    """
    Build an encoder (Vision Transformer) for I-JEPA.
    
    Args:
        model_name: one of 'vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge', 'vit_giant'
        img_size: input image size
        patch_size: patch size for patch embedding
        in_chans: number of input channels
        drop_path_rate: stochastic depth rate
        **kwargs: additional arguments passed to the model
        
    Returns:
        encoder: VisionTransformer model
        embed_dim: embedding dimension of the encoder
    """
    model_fn = {
        'vit_tiny': vit_tiny,
        'vit_small': vit_small,
        'vit_base': vit_base,
        'vit_large': vit_large,
        'vit_huge': vit_huge,
        'vit_giant': vit_giant,
    }
    
    if model_name not in model_fn:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_fn.keys())}")
    
    encoder = model_fn[model_name](
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    
    embed_dim = VIT_EMBED_DIMS[model_name]
    
    return encoder, embed_dim


def get_num_patches(img_size, patch_size):
    """
    Calculate the number of patches for a given image and patch size.
    
    Args:
        img_size: input image size (int or tuple)
        patch_size: patch size
        
    Returns:
        num_patches: total number of patches
    """
    if isinstance(img_size, (list, tuple)):
        img_size = img_size[0]
    return (img_size // patch_size) ** 2


# Re-export for convenience
__all__ = [
    'build_encoder',
    'get_num_patches',
    'VisionTransformer',
    'vit_tiny',
    'vit_small', 
    'vit_base',
    'vit_large',
    'vit_huge',
    'vit_giant',
    'VIT_EMBED_DIMS',
]
