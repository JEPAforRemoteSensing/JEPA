    # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Predictor module for I-JEPA.

The predictor is a narrow Vision Transformer that takes:
- Context encoder output (visible patches)
- Positional mask tokens for target positions

And predicts the representations at target positions.
"""

from functools import partial
import torch.nn as nn

from vision_transformer import VisionTransformerPredictor, vit_predictor


def build_predictor(
    num_patches,
    embed_dim=768,
    predictor_embed_dim=384,
    depth=6,
    num_heads=12,
    **kwargs
):
    """
    Build a predictor for I-JEPA.
    
    Args:
        num_patches: total number of patches in the image (needed for positional embeddings)
        embed_dim: embedding dimension of the encoder output
        predictor_embed_dim: internal embedding dimension of the predictor
        depth: number of transformer blocks in the predictor
        num_heads: number of attention heads
        **kwargs: additional arguments passed to the model
        
    Returns:
        predictor: VisionTransformerPredictor model
    """
    predictor = vit_predictor(
        num_patches=num_patches,
        embed_dim=embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=depth,
        num_heads=num_heads,
        **kwargs
    )
    
    return predictor


# Recommended predictor configurations for different encoder sizes
PREDICTOR_CONFIGS = {
    'vit_tiny': {
        'predictor_embed_dim': 96,
        'depth': 6,
        'num_heads': 3,
    },
    'vit_small': {
        'predictor_embed_dim': 192,
        'depth': 6,
        'num_heads': 6,
    },
    'vit_base': {
        'predictor_embed_dim': 384,
        'depth': 6,
        'num_heads': 12,
    },
    'vit_large': {
        'predictor_embed_dim': 512,
        'depth': 6,
        'num_heads': 16,
    },
    'vit_huge': {
        'predictor_embed_dim': 640,
        'depth': 12,
        'num_heads': 16,
    },
    'vit_giant': {
        'predictor_embed_dim': 704,
        'depth': 12,
        'num_heads': 16,
    },
}


def build_predictor_for_encoder(
    encoder_name,
    num_patches,
    embed_dim,
    **kwargs
):
    """
    Build a predictor with recommended configuration for a given encoder.
    
    Args:
        encoder_name: name of the encoder (e.g., 'vit_base')
        num_patches: total number of patches
        embed_dim: encoder embedding dimension
        **kwargs: override any default configuration
        
    Returns:
        predictor: VisionTransformerPredictor model
    """
    config = PREDICTOR_CONFIGS.get(encoder_name, PREDICTOR_CONFIGS['vit_base']).copy()
    config.update(kwargs)
    
    return build_predictor(
        num_patches=num_patches,
        embed_dim=embed_dim,
        **config
    )


# Re-export for convenience
__all__ = [
    'build_predictor',
    'build_predictor_for_encoder',
    'VisionTransformerPredictor',
    'vit_predictor',
    'PREDICTOR_CONFIGS',
]
