from typing import NamedTuple

import torch
from torch import nn

from src.wavelet import DWTLayer, IDWTLayer
from src.layer import BasicLayer
from .cnn import CNNRegularization

from timm.models.layers import trunc_normal_

class WaveValues(NamedTuple):
    xll: torch.Tensor
    xlh: torch.Tensor
    xhl: torch.Tensor
    xhh: torch.Tensor

class Transformer(nn.Module):
    def __init__(self, img_size=256, embed_dim=96, depths=[2], num_heads=[3],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.patch_norm = patch_norm

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim,
                               input_resolution=(img_size, img_size),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        out = self.norm(x).transpose(1,2).view(B, -1, H, W)
        return out

class WaveFormer(nn.Module):
    def __init__(self, wavename="haar", dim=96, in_chans=1, img_size=256, **kwargs):
        super().__init__()

        self.dwt = DWTLayer(wavename)
        self.idwt = IDWTLayer(wavename)

        self.conv_reg = CNNRegularization(in_chans=in_chans, n_convs=3, n_filters=dim, back_to_input=False)
        self.former_reg = Transformer(img_size=img_size // 2, embed_dim=dim)
        self.conv_out = nn.Conv2d(in_channels=2 * dim, out_channels=in_chans, kernel_size=5, padding=2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 0.01)

    def forward(self, x):
        # Embedding
        x_cnn = self.conv_reg(x)

        # Wavelet transform
        wavelet_values = WaveValues(*self.dwt(x_cnn))

        # Regularization
        xll_d = self.former_reg(wavelet_values.xll)
        x_idwt = self.idwt(xll_d, wavelet_values.xlh, wavelet_values.xhl, wavelet_values.xhh)
        x_reg = torch.cat([x_idwt, x_cnn], dim=1)

        # Output
        out = self.conv_out(x_reg)

        return out