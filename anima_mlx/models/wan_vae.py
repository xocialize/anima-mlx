"""Wan/Qwen-Image 16-ch 3D-causal VAE (decoder) in Python-MLX.

Transpose of the parity-validated Swift donor QwenVAE.swift. Channels-last
(B,T,H,W,C) internally; single-frame (T=1) image path (time_conv unused).
WanRMS = L2-over-channels norm (eps 1e-12), NOT mean-square. Module keys mirror
diffusers AutoencoderKLWan so weights load via convert_wan_vae_to_diffusers.
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn

LATENTS_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
LATENTS_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]


class CausalConv3d(nn.Module):
    def __init__(self, cin, cout, kernel=(3, 3, 3), pad=(1, 1)):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, kernel_size=kernel, padding=0)
        self.padT, self.padHW = pad

    def __call__(self, x):  # x: (B,T,H,W,C)
        if self.padT > 0 or self.padHW > 0:
            x = mx.pad(x, [(0, 0), (2 * self.padT, 0), (self.padHW, self.padHW),
                           (self.padHW, self.padHW), (0, 0)])
        return self.conv(x)


class WanRMSNorm(nn.Module):
    def __init__(self, channels, eps=1e-12):
        super().__init__()
        self.gamma = mx.ones((channels,))
        self.scale = channels ** 0.5
        self.eps = eps

    def __call__(self, x):  # channels-last, norm over last axis
        l2 = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        return x / mx.maximum(l2, mx.array(self.eps).astype(l2.dtype)) * self.scale * self.gamma


class WanResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.norm1 = WanRMSNorm(cin)
        self.conv1 = CausalConv3d(cin, cout, (3, 3, 3), (1, 1))
        self.norm2 = WanRMSNorm(cout)
        self.conv2 = CausalConv3d(cout, cout, (3, 3, 3), (1, 1))
        self.conv_shortcut = CausalConv3d(cin, cout, (1, 1, 1), (0, 0)) if cin != cout else None

    def __call__(self, x):
        res = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        return h + res


class WanAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = WanRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def __call__(self, x):  # (B,T,H,W,C)
        B, T, H, W, C = x.shape
        ident = x
        y = self.norm(x.reshape(B * T, H, W, C))
        qkv = self.to_qkv(y).reshape(B * T, H * W, 3, C)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        scale = 1.0 / (C ** 0.5)
        scores = mx.softmax(mx.matmul(q, k.transpose(0, 2, 1)) * scale, axis=-1)
        out = mx.matmul(scores, v).reshape(B * T, H, W, C)
        out = self.proj(out)
        return out.reshape(B, T, H, W, C) + ident


class WanUpsample(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        # resample.0 = Conv2d(dim -> dim//2); time_conv present for upsample3d (unused at T=1)
        self.resample = [nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)]
        self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), (1, 0)) if mode == "upsample3d" else None

    def __call__(self, x):  # (B,T,H,W,C) T=1 path: nearest-2x spatial + conv
        B, T, H, W, C = x.shape
        y = x.reshape(B * T, H, W, C)
        y = mx.repeat(mx.repeat(y, 2, axis=1), 2, axis=2)
        y = self.resample[0](y)
        return y.reshape(B, T, 2 * H, 2 * W, y.shape[-1])


class WanMidBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.resnets = [WanResBlock(dim, dim), WanResBlock(dim, dim)]
        self.attentions = [WanAttentionBlock(dim)]

    def __call__(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class WanUpBlock(nn.Module):
    def __init__(self, cin, cout, upsample_mode):
        super().__init__()
        blocks, dim = [], cin
        for _ in range(3):  # num_res_blocks + 1
            blocks.append(WanResBlock(dim, cout))
            dim = cout
        self.resnets = blocks
        self.upsamplers = [WanUpsample(cout, upsample_mode)] if upsample_mode else None

    def __call__(self, x):
        for r in self.resnets:
            x = r(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class WanVAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = CausalConv3d(16, 384, (3, 3, 3), (1, 1))
        self.mid_block = WanMidBlock(384)
        self.up_blocks = [
            WanUpBlock(384, 384, "upsample3d"),
            WanUpBlock(192, 384, "upsample3d"),
            WanUpBlock(192, 192, "upsample2d"),
            WanUpBlock(96, 96, None),
        ]
        self.norm_out = WanRMSNorm(96)
        self.conv_out = CausalConv3d(96, 3, (3, 3, 3), (1, 1))

    def __call__(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for b in self.up_blocks:
            x = b(x)
        return self.conv_out(nn.silu(self.norm_out(x)))


class WanVAE(nn.Module):
    """Decoder-only. decode() takes/returns PT layout (B,16,T,H,W)/(B,3,T,8H,8W)."""
    def __init__(self):
        super().__init__()
        self.post_quant_conv = CausalConv3d(16, 16, (1, 1, 1), (0, 0))
        self.decoder = WanVAEDecoder()

    def decode(self, latents):
        x = latents.transpose(0, 2, 3, 4, 1)        # (B,T,H,W,C)
        x = self.post_quant_conv(x)
        x = self.decoder(x)
        x = mx.clip(x, -1.0, 1.0)                     # diffusers decode() clamps to [-1,1]
        return x.transpose(0, 4, 1, 2, 3)            # (B,C,T,H,W)

    @staticmethod
    def denormalize(latents):
        mean = mx.array(LATENTS_MEAN).reshape(1, 16, 1, 1, 1)
        std = mx.array(LATENTS_STD).reshape(1, 16, 1, 1, 1)
        return latents * std + mean
