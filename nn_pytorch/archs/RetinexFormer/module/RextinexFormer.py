
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from einops import rearrange
from ..._shared.pad import pad, unpad



def _no_grad_trunc_normal_(
    tensor: Tensor,
    mean: float,
    std: float,
    a: float,
    b: float
) -> Tensor:
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor



def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.,
    std: float = 1.,
    a: float = -2.,
    b: float = 2.
) -> Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)



class GELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)



class Illumination_Estimator(nn.Module):
    def __init__(
        self,
        n_fea_middle: int,
        n_fea_in: int = 4,
        n_fea_out: int = 3,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle,
            n_fea_middle,
            kernel_size=5,
            padding=2,
            bias=True,
            groups=n_fea_in,
        )
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # x:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        mean_c = x.mean(dim=1).unsqueeze(1)
        input = torch.cat([x, mean_c], dim=1)

        x_1 = self.conv1(input)
        # illu_fea:   b,c,h,w
        illu_fea = self.depth_conv(x_1)
        # illu_map:   b,c=3,h,w
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map



class IG_MSA(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in: Tensor, illu_fea_trans) -> Tensor:
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp: Tensor = self.to_q(x)
        k_inp: Tensor = self.to_k(x)
        v_inp: Tensor = self.to_v(x)
        # illu_fea: b,c,h,w -> b,h,w,c
        illu_attn = illu_fea_trans
        q, k, v, illu_attn = (
            rearrange(t, "b n (h d) -> b h n d", h=self.num_heads)
            for t in (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2))
        )
        v = v * illu_attn

        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        # A = K^T*Q
        attn = k @ q.transpose(-2, -1)
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)

        # b,heads,d,hw
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)

        out_c = self.proj(x).view(b, h, w, c)
        out_p = (
            self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2))
            .permute(0, 2, 3, 1)
        )

        return out_c + out_p



class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out: Tensor = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)



class IGAB(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList([
                    IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                    PreNorm(dim, FeedForward(dim=dim)),
                ])
            )

    def forward(self, x: Tensor, illu_fea: Tensor) -> Tensor:
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out



class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super().__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList([
                    IGAB(
                        dim=dim_level,
                        num_blocks=num_blocks[i],
                        dim_head=dim,
                        heads=dim_level // dim,
                    ),
                    nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                    nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                ])
            )
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level,
            dim_head=dim,
            heads=dim_level // dim,
            num_blocks=num_blocks[-1],
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(
                        dim_level,
                        dim_level // 2,
                        stride=2,
                        kernel_size=2,
                        padding=0,
                        output_padding=0,
                    ),
                    nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                    IGAB(
                        dim=dim_level // 2,
                        num_blocks=num_blocks[level - 1 - i],
                        dim_head=dim,
                        heads=(dim_level // 2) // dim,
                    ),
                ])
            )
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x: Tensor, illu_fea: Tensor) -> Tensor:
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """
        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for IGAB, FeaDownSample, IlluFeaDownsample in self.encoder_layers:
            # bchw
            fea = IGAB(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea, illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, illu_fea)

        # Mapping
        return self.mapping(fea) + x



class RetinexFormer_Single_Stage(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        n_feat=31,
        level=2,
        num_blocks=[1, 1, 1],
    ):
        super().__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(
            in_dim=in_channels,
            out_dim=out_channels,
            dim=n_feat,
            level=level,
            num_blocks=num_blocks,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x:        b,c=3,h,w
        # illu_fea: b,c,h,w
        # illu_map: b,c=3,h,w
        illu_fea, illu_map = self.estimator(x)
        input_img = x * illu_map + x
        return self.denoiser(input_img, illu_fea)



class RetinexFormer(nn.Module):

    def __init__(
        self,
        *,
        in_nc: int = 3,
        out_nc: int = 3,
        n_feat: int = 40,
        stage: int = 3,
        num_blocks: list[int] = [1, 1, 1],
    ):
        super().__init__()
        self.stage = stage

        modules_body = [
            RetinexFormer_Single_Stage(
                in_channels=in_nc,
                out_channels=out_nc,
                n_feat=n_feat,
                level=2,
                num_blocks=num_blocks,
            )
            for _ in range(stage)
        ]

        self.body = nn.Sequential(*modules_body)


    def forward(self, x: Tensor) -> Tensor:
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        size = x.shape[2:]
        x = pad(x, modulo=8, mode='reflect')
        out: Tensor = self.body(x)
        out = unpad(out, size, scale=1)

        return torch.clamp(out, 0., 1.)
