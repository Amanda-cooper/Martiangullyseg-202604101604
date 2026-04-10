# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch, yaml
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os
import numpy as np

__all__ = ['fasternet_t0', 'fasternet_t1', 'fasternet_t2', 'fasternet_s', 'fasternet_m', 'fasternet_l']

class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):

        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchMerging(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x


class FasterNet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(1, 2, 8, 2),
                 mlp_ratio=2.,
                 n_div=4,
                 patch_size=4,
                 patch_stride=4,
                 patch_size2=2,  # for subsequent layers
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 init_cfg=None,
                 pretrained=None,
                 pconv_fw_type='split_cat',
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                               n_div=n_div,
                               depth=depths[i_stage],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type
                               )
            stages_list.append(stage)

            # patch merging layer
            if i_stage < self.num_stages - 1:
                stages_list.append(
                    PatchMerging(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** i_stage),
                                 norm_layer=norm_layer)
                )

        self.stages = nn.Sequential(*stages_list)

        # add a norm layer for each output
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                raise NotImplementedError
            else:
                layer = norm_layer(int(embed_dim * 2 ** i_emb))
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
    def forward(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs

def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict

class FasterNet_t0Wrapper(nn.Module):
    """FasterNet_t0包装类，用于UNet集成"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # 获取FasterNet_t0的输出
        feats = self.model(x)
        
        # FasterNet_t0输出4个特征图，UNet需要5个
        feat1, feat2, feat3, feat4 = feats
        
        # 获取输入尺寸
        input_h, input_w = x.shape[2], x.shape[3]
        
        # 调整空间尺寸以匹配VGG的输出模式
        # VGG输出: [input_size, input_size/2, input_size/4, input_size/8, input_size/16]
        # FasterNet_t0输出: [input_size/4, input_size/8, input_size/16, input_size/32]
        
        # 将feat1从input_size/4上采样到input_size
        feat1 = torch.nn.functional.interpolate(feat1, size=(input_h, input_w), mode='bilinear', align_corners=False)
        
        # 将feat2从input_size/8上采样到input_size/2
        feat2 = torch.nn.functional.interpolate(feat2, size=(input_h//2, input_w//2), mode='bilinear', align_corners=False)
        
        # 将feat3从input_size/16上采样到input_size/4
        feat3 = torch.nn.functional.interpolate(feat3, size=(input_h//4, input_w//4), mode='bilinear', align_corners=False)
        
        # 将feat4从input_size/32上采样到input_size/8
        feat4 = torch.nn.functional.interpolate(feat4, size=(input_h//8, input_w//8), mode='bilinear', align_corners=False)
        
        # 添加第5个特征图，使用feat4
        feat5 = torch.nn.functional.interpolate(feat4, size=(input_h//16, input_w//16), mode='bilinear', align_corners=False)
        
        return [feat1, feat2, feat3, feat4, feat5]

def fasternet_t0(weights=None, cfg='block/backbone/faster_cfg/fasternet_t0.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    
    # 返回包装后的模型
    return FasterNet_t0Wrapper(model)

def fasternet_t1(weights=None, cfg='ultralytics/nn/backbone/faster_cfg/fasternet_t1.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def fasternet_t2(weights=None, cfg='ultralytics/nn/backbone/faster_cfg/fasternet_t2.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def fasternet_s(weights=None, cfg='ultralytics/nn/backbone/faster_cfgg/fasternet_s.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def fasternet_m(weights=None, cfg='ultralytics/nn/backbone/faster_cfg/fasternet_m.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def fasternet_l(weights=None, cfg='ultralytics/nn/backbone/faster_cfg/fasternet_l.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model


class FasterNetBackbone(nn.Module):
    """
    把 FasterNet 改造成 DeepLab 需要的双输出接口：
        low_level_features: 1/4  分辨率，对应 stage 1 输出
        x               : 1/16 分辨率，对应 stage 3 输出
    downsample_factor 仅支持 16（如需 8 需额外改动）
    """
    def __init__(self, downsample_factor=16):
        super().__init__()
        assert downsample_factor == 16, \
            "FasterNet 默认 4×8×16×32 下采样，仅支持 16"
        # 这里以 t0 为例，可按需换 t1/t2/s/m/l
        self.backbone = fasternet_t0()
        # FasterNet 返回 4 层特征 [1/4, 1/8, 1/16, 1/32]
        self.low_level_idx  = 0   # 1/4
        self.high_level_idx = 2   # 1/16

    def forward(self, x):
        feats = self.backbone(x)            # list: [s1,s2,s3,s4]
        low_level_features = feats[self.low_level_idx]   # [B, C1, H/4, W/4]
        x = feats[self.high_level_idx]                   # [B, C3, H/16, W/16]
        return low_level_features, x



if __name__ == '__main__':
    import yaml
    model = fasternet_t0(weights='fasternet_t0-epoch.281-val_acc1.71.9180.pth', cfg='cfg/fasternet_t0.yaml')
    print(model.channel)
    inputs = torch.randn((1, 3, 640, 640))
    for i in model(inputs):
        print(i.size())