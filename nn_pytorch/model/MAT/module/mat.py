import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from timm.layers import to_2tuple

from .basic_module import (
    FullyConnectedLayer,
    Conv2dLayer,
    MappingNet,
    StyleConv,
    ToRGB,
    get_style_code
)


def nf(stage, channel_base=32768, channel_decay=1.0, channel_max=512):
    NF = {
        512: 64,
        256: 128,
        128: 256,
        64: 512,
        32: 512,
        16: 512,
        8: 512,
        4: 512
    }
    return NF[2 ** stage]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = FullyConnectedLayer(in_features=in_features, out_features=hidden_features, activation='lrelu')
        self.fc2 = FullyConnectedLayer(in_features=hidden_features, out_features=out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x



def window_partition(x: Tensor, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows



def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class Conv2dLayerPartial(nn.Module):
    def __init__(
        self,
        in_channels: int,   # Number of input channels.
        out_channels: int,  # Number of output channels.
        kernel_size: int,   # Width and height of the convolution kernel.
        bias: bool = True,  # Apply additive bias before the activation function?
        activation: str = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up = 1,             # Integer upsampling factor.
        down = 1,           # Integer downsampling factor.
        resample_filter = [1, 3, 3, 1],    # Low-pass filter to apply when resampling activations.
        conv_clamp = None,  # Clamp the output to +-X, None = disable clamping.
        trainable = True,   # Update the weights of this layer during training?
    ):
        super().__init__()
        self.conv = Conv2dLayer(
            in_channels,
            out_channels,
            kernel_size,
            bias,
            activation,
            up,
            down,
            resample_filter,
            conv_clamp,
            trainable
        )

        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_winsize = kernel_size ** 2
        self.stride = down
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0

    def forward(self, x: Tensor, mask=None):
        if mask is not None:
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding
                )
                # mask_ratio = self.slide_winsize / (update_mask + torch.finfo(x.dtype).eps)
                mask_ratio = self.slide_winsize / (update_mask.to(torch.float32) + 1e-8)
                update_mask = torch.clamp_(update_mask, 0, 1)
                mask_ratio = torch.mul(mask_ratio, update_mask).to(x.dtype)
            x = self.conv(x)
            x = torch.mul(x, mask_ratio)
            return x, update_mask
        else:
            x = self.conv(x)
            return x, None



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int],
        num_heads: int,
        down_ratio: int = 1,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.k = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.v = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.proj = FullyConnectedLayer(in_features=dim, out_features=dim)

        self.softmax = nn.Softmax(dim=-1)


    def forward(
        self,
        x: Tensor,
        mask_windows: Tensor | None = None,
        mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        norm_x = F.normalize(x, p=2.0, dim=-1, eps=torch.finfo(x.dtype).eps)
        q = (
            self.q(norm_x)
            .reshape(B_, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(norm_x)
            .view(B_, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.v(x)
            .view(B_, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = (
                attn.view(B_ // nW, nW, self.num_heads, N, N)
                + mask.unsqueeze(1).unsqueeze(0)
            )
            attn = attn.view(-1, self.num_heads, N, N)

        if mask_windows is not None:
            attn_mask_windows = mask_windows.squeeze(-1).unsqueeze(1).unsqueeze(1)
            attn = (
                attn
                + attn_mask_windows.masked_fill(
                    attn_mask_windows == 0, float(-100.0)
                ).masked_fill(attn_mask_windows == 1, float(0.0))
            )
            with torch.no_grad():
                mask_windows = torch.sum(mask_windows, dim=1, keepdim=True)
                mask_windows = torch.clamp_(mask_windows, 0, 1)
                mask_windows = mask_windows.repeat(1, N, 1)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x, mask_windows



class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int],
        num_heads: int,
        down_ratio: int = 1,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if self.shift_size > 0:
            down_ratio = 1
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            down_ratio=down_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.fuse = FullyConnectedLayer(
            in_features=dim * 2, out_features=dim, activation='lrelu'
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)


    def calculate_mask(self, x_size: tuple[int, int]):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # nW, window_size, window_size, 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)
        )

        return attn_mask


    def forward(self, x: Tensor, x_size: tuple[int, int], mask: Tensor | None = None):
        # H, W = self.input_resolution
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)
        if mask is not None:
            mask = mask.view(B, H, W, 1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            if mask is not None:
                shifted_mask = torch.roll(
                    mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
                )
        else:
            shifted_x = x
            if mask is not None:
                shifted_mask = mask

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        if mask is not None:
            mask_windows = window_partition(shifted_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
        else:
            mask_windows = None

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            # nW*B, window_size*window_size, C
            attn_windows, mask_windows = self.attn(
                x_windows, mask_windows, mask=self.attn_mask
            )
        else:
            # nW*B, window_size*window_size, C
            attn_windows, mask_windows = self.attn(
                x_windows,
                mask_windows,
                mask=self.calculate_mask(x_size).to(dtype=x.dtype).to(x.device)
            )

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # B H' W' C
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if mask is not None:
            mask_windows = mask_windows.view(-1, self.window_size, self.window_size, 1)
            shifted_mask = window_reverse(mask_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
            if mask is not None:
                mask = torch.roll(
                    shifted_mask, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
                )
        else:
            x = shifted_x
            if mask is not None:
                mask = shifted_mask
        x = x.view(B, H * W, C)
        if mask is not None:
            mask = mask.view(B, H * W, 1)

        # FFN
        x = self.fuse(torch.cat([shortcut, x], dim=-1))
        x = self.mlp(x)

        return x, mask



class PatchMerging(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, down: int = 2):
        super().__init__()
        self.conv = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation='lrelu',
            down=down,
        )
        self.down = down

    def forward(self, x: Tensor, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.down != 1:
            ratio = 1 / self.down
            x_size = (int(x_size[0] * ratio), int(x_size[1] * ratio))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask



class PatchUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, up=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation='lrelu',
            up=up,
        )
        self.up = up

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.up != 1:
            x_size = (int(x_size[0] * self.up), int(x_size[1] * self.up))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask



class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int],
        depth: int,
        num_heads: int,
        window_size: int,
        down_ratio: int = 1,
        mlp_ratio: float = 2.,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        if downsample is not None:
            # self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.downsample = downsample
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, down_ratio=down_ratio, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.conv = Conv2dLayerPartial(
            in_channels=dim, out_channels=dim, kernel_size=3, activation='lrelu'
        )


    def forward(self, x: Tensor, x_size, mask=None) -> Tensor:
        if self.downsample is not None:
            x, x_size, mask = self.downsample(x, x_size, mask)
        identity = x
        for blk in self.blocks:
            if self.use_checkpoint:
                x, mask = checkpoint.checkpoint(blk, x, x_size, mask)
            else:
                x, mask = blk(x, x_size, mask)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(token2feature(x, x_size), mask)
        x = feature2token(x) + identity
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask



class ToToken(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 128,
        kernel_size: int = 5,
        stride: int = 1
    ):
        super().__init__()
        self.proj = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=kernel_size,
            activation='lrelu'
        )

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        x, mask = self.proj(x, mask)

        return x, mask



class EncFromRGB(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str):
        # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
        )
        self.conv1 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv0(x)
        x = self.conv1(x)
        return x



class ConvBlockDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str):
        # res = 2, ..., resolution_log
        super().__init__()
        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
            down=2,
        )
        self.conv1 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv0(x)
        x = self.conv1(x)
        return x



def token2feature(x: Tensor, x_size) -> Tensor:
    B, N, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x



def feature2token(x: Tensor) -> Tensor:
    B, C, H, W = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x



class Encoder(nn.Module):
    def __init__(
        self,
        res_log2: int,
        img_channels: int,
        activation: str,
        patch_size: int = 5,
        channels: int = 16,
        drop_path_rate: float = 0.1
    ):
        super().__init__()

        self.resolution = []
        for idx, i in enumerate(range(res_log2, 3, -1)):
            # from input size to 16x16
            res = 2 ** i
            self.resolution.append(res)
            if i == res_log2:
                block = EncFromRGB(img_channels * 2 + 1, nf(i), activation)
            else:
                block = ConvBlockDown(nf(i+1), nf(i), activation)
            setattr(self, 'EncConv_Block_%dx%d' % (res, res), block)


    def forward(self, x: Tensor) -> dict[int, Tensor]:
        out = {}
        for res in self.resolution:
            res_log2 = int(np.log2(res))
            x = getattr(self, 'EncConv_Block_%dx%d' % (res, res))(x)
            out[res_log2] = x

        return out



class ToStyle(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        drop_rate
    ):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
            Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
            Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
            )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = FullyConnectedLayer(
            in_features=in_channels,
            out_features=out_channels,
            activation=activation
        )
        # self.dropout = nn.Dropout(drop_rate)


    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))
        # x = self.dropout(x)
        return x



class DecBlockFirstV2(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.res = res

        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            activation=activation,
        )
        self.conv1 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )


    def forward(self, x: Tensor, ws, gs, E_features, noise_mode: str='random'):
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img



class DecBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):  # res = 4, ..., resolution_log2
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, img, ws, gs, E_features, noise_mode='random'):
        style = get_style_code(ws[:, self.res * 2 - 9], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 8], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 7], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img



class Decoder(nn.Module):
    def __init__(self, res_log2, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.Dec_16x16 = DecBlockFirstV2(4, nf(4), nf(4), activation, style_dim, use_noise, demodulate, img_channels)
        for res in range(5, res_log2 + 1):
            setattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res),
                    DecBlock(res, nf(res - 1), nf(res), activation, style_dim, use_noise, demodulate, img_channels))
        self.res_log2 = res_log2

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        x, img = self.Dec_16x16(x, ws, gs, E_features, noise_mode=noise_mode)
        for res in range(5, self.res_log2 + 1):
            block = getattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res))
            x, img = block(x, img, ws, gs, E_features, noise_mode=noise_mode)

        return img



class DecStyleBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, img, style, skip, noise_mode='random'):
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + skip
        x = self.conv1(x, style, noise_mode=noise_mode)
        img = self.toRGB(x, style, skip=img)

        return x, img



class FirstStage(nn.Module):
    def __init__(
        self,
        img_channels: int,
        img_resolution: int = 256,
        dim: int = 180,
        w_dim: int = 512,
        use_noise: bool = False,
        demodulate: bool = True,
        activation: str = 'lrelu'
    ):
        super().__init__()
        res = 64

        self.conv_first = Conv2dLayerPartial(
            in_channels=img_channels+1,
            out_channels=dim,
            kernel_size=3,
            activation=activation
        )
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))
        # from input size to 64
        for i in range(down_time):
            self.enc_conv.append(
                Conv2dLayerPartial(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    down=2,
                    activation=activation
                )
            )

        # from 64 -> 16 -> 64
        depths = [2, 3, 4, 3, 2]
        ratios = [1, 1/2, 1/2, 2, 2]
        num_heads = 6
        window_sizes = [8, 16, 16, 16, 8]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.tran = nn.ModuleList()
        for i, depth in enumerate(depths):
            res = int(res * ratios[i])
            if ratios[i] < 1:
                merge = PatchMerging(dim, dim, down=int(1/ratios[i]))
            elif ratios[i] > 1:
                merge = PatchUpsampling(dim, dim, up=ratios[i])
            else:
                merge = None
            self.tran.append(
                BasicLayer(
                    dim=dim,
                    input_resolution=[res, res],
                    depth=depth,
                    num_heads=num_heads,
                    window_size=window_sizes[i],
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    downsample=merge
                )
            )

        # global style
        down_conv = []
        for i in range(int(np.log2(16))):
            down_conv.append(
                Conv2dLayer(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    down=2,
                    activation=activation
                )
            )
        down_conv.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.down_conv = nn.Sequential(*down_conv)
        self.to_style = FullyConnectedLayer(
            in_features=dim,
            out_features=dim*2,
            activation=activation
        )
        self.ws_style = FullyConnectedLayer(
            in_features=w_dim,
            out_features=dim,
            activation=activation
        )
        self.to_square = FullyConnectedLayer(
            in_features=dim,
            out_features=16*16,
            activation=activation
        )

        style_dim = dim * 3
        self.dec_conv = nn.ModuleList()
        # from 64 to input size
        for i in range(down_time):
            res = res * 2
            self.dec_conv.append(
                DecStyleBlock(
                    res,
                    dim,
                    dim,
                    activation,
                    style_dim,
                    use_noise,
                    demodulate,
                    img_channels
                )
            )

    def forward(self, images_in, masks_in, ws, noise_mode='random'):
        x: Tensor = torch.cat([masks_in - 0.5, images_in * masks_in], dim=1)

        skips = []
        # input size
        x, mask = self.conv_first(x, masks_in)
        skips.append(x)
        # input size to 64
        for i, block in enumerate(self.enc_conv):
            x, mask = block(x, mask)
            if i != len(self.enc_conv) - 1:
                skips.append(x)

        x_size = x.size()[-2:]
        x = feature2token(x)
        mask = feature2token(mask)
        mid = len(self.tran) // 2
        # 64 to 16
        for i, block in enumerate(self.tran):
            if i < mid:
                x, x_size, mask = block(x, x_size, mask)
                skips.append(x)
            elif i > mid:
                x, x_size, mask = block(x, x_size, None)
                x = x + skips[mid - i]
            else:
                x, x_size, mask = block(x, x_size, None)
                mul_map = torch.ones_like(x, dtype=x.dtype) * 0.5
                mul_map = F.dropout(mul_map, training=True)
                ws = self.ws_style(ws[:, -1])
                add_n = self.to_square(ws).unsqueeze(1)
                add_n = F.interpolate(
                    add_n,
                    size=x.size(1),
                    mode='linear',
                    align_corners=False
                ).squeeze(1).unsqueeze(-1)
                x = x * mul_map + add_n * (1 - mul_map)
                gs = self.to_style(
                    self.down_conv(token2feature(x, x_size)).flatten(start_dim=1)
                )
                style = torch.cat([gs, ws], dim=1)

        x = token2feature(x, x_size).contiguous()
        img = None
        for i, block in enumerate(self.dec_conv):
            x, img = block(
                x,
                img,
                style,
                skips[len(self.dec_conv)-i-1],
                noise_mode=noise_mode
            )

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        return img



class SynthesisNet(nn.Module):
    def __init__(
        self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels: int = 3,      # Number of color channels.
        channel_base: int = 32768,  # Overall multiplier for the number of channels.
        channel_decay: float= 1.0,
        channel_max: int = 512,     # Maximum number of channels in any layer.
        activation: str = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        drop_rate: float= 0.5,
        use_noise: bool = False,
        demodulate: bool = True,
    ):
        super().__init__()
        resolution_log2 = int(np.log2(img_resolution))
        # assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4

        self.num_layers = resolution_log2 * 2 - 3 * 2
        self.img_resolution = img_resolution
        self.resolution_log2 = resolution_log2

        # first stage
        self.first_stage = FirstStage(
            img_channels,
            img_resolution=img_resolution,
            w_dim=w_dim,
            use_noise=False,
            demodulate=demodulate
        )

        # second stage
        self.enc = Encoder(
            resolution_log2,
            img_channels,
            activation,
            patch_size=5,
            channels=16
        )
        self.to_square = FullyConnectedLayer(
            in_features=w_dim,
            out_features=16*16,
            activation=activation
        )
        self.to_style = ToStyle(
            in_channels=nf(4),
            out_channels=nf(2) * 2,
            activation=activation,
            drop_rate=drop_rate
        )
        style_dim = w_dim + nf(2) * 2
        self.dec = Decoder(
            resolution_log2,
            activation,
            style_dim,
            use_noise,
            demodulate,
            img_channels
        )


    def forward(
        self,
        images_in,
        masks_in,
        ws,
        noise_mode='random'
    ):
        out_stg1 = self.first_stage(images_in, masks_in, ws, noise_mode=noise_mode)

        # encoder
        x = images_in * masks_in + out_stg1 * (1 - masks_in)
        x = torch.cat([masks_in - 0.5, x, images_in * masks_in], dim=1)
        E_features = self.enc(x)

        fea_16 = E_features[4]
        mul_map = torch.ones_like(fea_16) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        add_n = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1)
        add_n = F.interpolate(
            add_n, size=fea_16.size()[-2:], mode='bilinear', align_corners=False
        )
        fea_16 = fea_16 * mul_map + add_n * (1 - mul_map)
        E_features[4] = fea_16

        # style
        gs = self.to_style(fea_16)

        # decoder
        img = self.dec(fea_16, ws, gs, E_features, noise_mode=noise_mode)

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        return img



class Generator(nn.Module):
    def __init__(
        self,
        z_dim: float,
        c_dim,
        w_dim,
        img_resolution,
        img_channels: int = 3,
        synthesis_kwargs: dict = {},
        mapping_kwargs: dict = {},
    ):
        """
            z_dim           Input latent (Z) dimensionality, 0 = no latent.
            c_dim           Conditioning label (C) dimensionality, 0 = no label.
            w_dim           Intermediate latent (W) dimensionality.
            img_resolution  resolution of generated image
            img_channels    Number of input color channels.
            synthesis_kwargs Arguments for SynthesisNetwork.
            mapping_kwargs  Arguments for MappingNetwork.
        """
        super().__init__()
        self.z_dim: float = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.synthesis = SynthesisNet(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs
        )
        self.mapping = MappingNet(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            num_ws=self.synthesis.num_layers,
            **mapping_kwargs
        )


    def forward(
        self,
        images_in,
        masks_in,
        z,
        c,
        truncation_psi: float = 1.,
        truncation_cutoff = None,
        skip_w_avg_update = False,
        noise_mode='random',
    ):
        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            skip_w_avg_update=skip_w_avg_update
        )

        return self.synthesis(images_in, masks_in, ws, noise_mode=noise_mode)



class MAT(nn.Module):

    def __init__(self, fp16: bool = True):
        super().__init__()
        self.model = Generator(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=512,
            img_channels=3,
        )
        self.z_dim: int = self.model.z_dim
        self.c_dim: int = self.model.c_dim


    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # Pad both img and mask
        # n, w, h, c
        # w = x.shape[3] if x.shape[1] <= 3 else x.shape[1]
        # n, c, h, w
        h, w = x.shape[2:]

        mod_pad_h: int = (512 - h % 512) % 512
        mod_pad_w: int = (512 - w % 512) % 512
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant", 0).contiguous()
        mask = F.pad(mask, (0, mod_pad_w, 0, mod_pad_h), "constant", 0).contiguous()

        # [0, 1] -> [-1, 1]
        x = x * 2 - 1
        mask = 1 - mask

        device: str = x.device
        dtype: torch.dtype = x.dtype
        label = torch.zeros([1, self.c_dim], device=device, dtype=dtype)
        z: Tensor = (
            torch.from_numpy(np.random.randn(1, self.z_dim))
            .to(device=device, dtype=dtype)
        )

        out: Tensor = self.model(
            x,
            mask,
            z,
            label,
            truncation_psi=1.,
            noise_mode="none"
        )

        if mod_pad_h or mod_pad_w:
            out = out[:, :, :h, :w]
        return (out + 1) / 2

