# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

from utils import trunc_normal_, repeat_interleave_batch, apply_masks


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sinusoidal positional embeddings.
    
    Args:
        embed_dim: embedding dimension
        grid_size: int of the grid height and width
        cls_token: whether to include cls token position
        
    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 1D sinusoidal positional embeddings.
    
    Args:
        embed_dim: embedding dimension
        grid_size: int of the grid length
        cls_token: whether to include cls token position
        
    Returns:
        pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim]
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sinusoidal positional embeddings from grid positions.
    
    Args:
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        
    Returns:
        emb: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    """MLP block with GELU activation."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
    
class CrossAttention(nn.Module):
    """Multi-head cross-attention module."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        # Q from x (query)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, _, _ = qkv[0], qkv[1], qkv[2]
        
        # K, V from context (keys and values)
        B_ctx, N_ctx, C_ctx = context.shape
        qkv_context = self.qkv(context).reshape(B_ctx, N_ctx, 3, self.num_heads, C_ctx // self.num_heads).permute(2, 0, 3, 1, 4)
        _, k_context, v_context = qkv_context[0], qkv_context[1], qkv_context[2]

        attn = (q @ k_context.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v_context).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class CrossBlock(nn.Module):
    """Transformer block with cross-attention and MLP."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, context, return_attention=False):
        y, attn = self.attn(self.norm1(x), context)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding using Conv2d."""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for I-JEPA.
    
    This serves as both the context encoder and target encoder.
    The target encoder uses an exponential moving average (EMA) of the context encoder weights.
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Handle img_size as list or int
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding (sinusoidal, frozen)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        
        # Weight initialization
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        """Rescale weights for better initialization."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None):
        """
        Forward pass through the encoder.
        
        Args:
            x: input images of shape (B, C, H, W)
            masks: optional list of mask tensors for context masking
            
        Returns:
            Patch-level representations of shape (B, N, D) or masked version
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # Patchify input
        x = self.patch_embed(x)
        B, N, D = x.shape

        # Add positional embedding
        x = x + self.pos_embed

        # Apply masks if provided (for context encoder)
        if masks is not None:
            x = apply_masks(x, masks)

        # Forward through transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Final normalization
        if self.norm is not None:
            x = self.norm(x)

        return x


class VisionTransformerPredictor(nn.Module):
    """
    Vision Transformer Predictor for I-JEPA.
    
    Takes context encoder output and predicts target block representations.
    Uses mask tokens with positional embeddings to specify which positions to predict.
    """
    
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        
        # Project from encoder dimension to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Positional embedding for predictor (sinusoidal, frozen)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1],
            int(num_patches ** 0.5),
            cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        
        # Predictor transformer blocks
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.predictor_norm = norm_layer(predictor_embed_dim)
        
        # Project back to encoder dimension
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        
        # Weight initialization
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        """Rescale weights for better initialization."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks):
        """
        Forward pass through the predictor.
        
        Args:
            x: context encoder output of shape (B*nenc, N_ctx, D)
            masks_x: list of context masks (encoder masks)
            masks: list of target masks (prediction masks)
            
        Returns:
            Predicted representations for target positions
        """
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # Batch size (accounting for multiple encoder masks)
        B = len(x) // len(masks_x)

        # Map from encoder-dim to predictor-dim
        x = self.predictor_embed(x)

        # Add positional embedding to context tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # Create mask tokens with positional embeddings for target positions
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        # pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        
        # Initialize prediction tokens
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs
        
        # Concatenate context tokens with prediction tokens
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # Forward through predictor blocks
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # Return only predictions for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x


class SharedPredictor(nn.Module):
    """
    Vision Transformer Predictor for I-JEPA.
    
    Takes context encoder output and predicts target block representations.
    Uses mask tokens with positional embeddings to specify which positions to predict.
    """
    
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=768,
        depth=1,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        num_tokens=4,
        **kwargs
    ):
        super().__init__()
        
        # Project from encoder dimension to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.num_tokens = num_tokens
        self.learnable_tokens = nn.Parameter(torch.zeros(1, num_tokens, predictor_embed_dim))
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Positional embedding for predictor (sinusoidal, frozen)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1],
            int(num_patches ** 0.5),
            cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        
        # Predictor transformer blocks
        self.predictor_self_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.predictor_cross_blocks = nn.ModuleList([
            CrossBlock(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.predictor_norm = norm_layer(predictor_embed_dim)
        
        # Project back to encoder dimension
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        
        # Weight initialization
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.learnable_tokens, std=self.init_std)  # Initialize learnable tokens!
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        """Rescale weights for better initialization."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_self_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
        
        for layer_id, layer in enumerate(self.predictor_cross_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_ctx, masks_tgt, other_ctx):
        """
        Forward pass through the predictor.
        
        Args:
            x: context encoder output of shape (B*nenc, N_ctx, D)
            masks_ctx: list of context masks (encoder masks)
            masks_tgt: list of target masks (prediction masks)
            other_ctx: cross-attention context from other modality
            
        Returns:
            Predicted representations for target positions (B, N_pred, D)
            Same shape as without learnable queries - queries participate in 
            attention but are excluded from final output.
        """
        assert (masks_tgt is not None) and (other_ctx is not None), 'Cannot run predictor without mask indices'

        # Batch size (accounting for multiple encoder masks)
        B = len(x) // len(masks_ctx)

        # Map from encoder-dim to predictor-dim
        x = self.predictor_embed(x)

        # Add positional embedding to context tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_ctx)

        _, N_ctxt, D = x.shape

        # Create mask tokens with positional embeddings for target positions
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks_tgt)
        
        # Initialize prediction tokens
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs
        N_pred = pred_tokens.size(1)
        
        # Expand learnable tokens for batch (use contiguous to avoid view/stride issues)
        learnable_queries = self.learnable_tokens.expand(B * len(masks_tgt), -1, -1).contiguous()
        
        # Concatenate: [context, pred_tokens, learnable_queries]
        # Learnable queries at the end so we can easily slice them off
        x = x.repeat(len(masks_tgt), 1, 1)
        x = torch.cat([x, pred_tokens, learnable_queries], dim=1)

        # Forward through predictor blocks
        # Learnable queries participate in self-attention, influencing all tokens
        for self_blk, cross_blk in zip(self.predictor_self_blocks, self.predictor_cross_blocks):
            x = self_blk(x)
            x = cross_blk(x, other_ctx)
        x = self.predictor_norm(x)

        # Return only prediction tokens (exclude context and learnable queries)
        # x shape: [B, N_ctxt + N_pred + num_tokens, D]
        # We want: [B, N_pred, D]
        x = x[:, N_ctxt:N_ctxt + N_pred, :]
        x = self.predictor_proj(x)

        return x


# ============================================================================
# Factory functions for creating models
# ============================================================================

def vit_predictor(**kwargs):
    """Create a VisionTransformerPredictor."""
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_tiny(patch_size=16, **kwargs):
    """ViT-Tiny: 192 dim, 12 layers, 3 heads."""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    """ViT-Small: 384 dim, 12 layers, 6 heads."""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    """ViT-Base: 768 dim, 12 layers, 12 heads."""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    """ViT-Large: 1024 dim, 24 layers, 16 heads."""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    """ViT-Huge: 1280 dim, 32 layers, 16 heads."""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    """ViT-Giant: 1408 dim, 40 layers, 16 heads."""
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# Embedding dimensions for each model variant
VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}
