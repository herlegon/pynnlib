import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import einsum, nn

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        _N, C, _H, _W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        return weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        _N, C, _H, _W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


def pixelshuffle_block(
    in_channels, out_channels, upscale_factor=2, kernel_size=3, bias=False
):
    """
    Upsample features according to `upscale_factor`.
    """
    kernel_size // 2
    conv = nn.Conv2d(
        in_channels,
        out_channels * (upscale_factor**2),
        kernel_size,
        padding=1,
        bias=bias,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
        super().__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(
            c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False
        )
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=1, bias=False):
        super().__init__()

        hidden_features = int(dim * mult)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )

    def forward(self, x):
        return x * self.gate(x)


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0.0 or (not self.training):
            return x

        keep_mask = (
            torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_()
            > self.prob
        )
        return x * keep_mask / (1 - self.prob)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


def MBConv(
    dim_in, dim_out, *, downsample, expansion_rate=4, shrinkage_rate=0.25, dropout=0.0
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.GELU(),
        nn.Conv2d(
            hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim
        ),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


class LB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(
        self, dim_in, dim_out, downsample=False, expansion_rate=1, shrinkage_rate=0.25
    ):
        super().__init__()

        self.net = MBConv(
            dim_in,
            dim_out,
            downsample=downsample,
            expansion_rate=expansion_rate,
            shrinkage_rate=shrinkage_rate,
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionA(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0.0, window_size=7, with_pe=True):
        super().__init__()
        assert (dim % dim_head) == 0, (
            "dimension should be divisible by dimension per head"
        )
        self.heads = dim // dim_head
        self.scale = dim_head**-0.5
        self.with_pe = with_pe
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Dropout(dropout)
        )
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos))
            grid = rearrange(grid, "c i j -> (i j) c")
            rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(
                grid, "j ... -> 1 j ..."
            )
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(
                dim=-1
            )

            self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        _batch, height, width, window_height, window_width, _, _device, h = (
            *x.shape,
            x.device,
            self.heads,
        )
        # flatten
        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")
        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # split heads
        q, k, v = (rearrange(t, "b n (h d ) -> b h n d", h=h) for t in (q, k, v))
        # scale
        q = q * self.scale
        # sim
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, "i j h -> h i j")
        # attention
        attn = self.attend(sim)
        # aggregate
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        # merge heads
        out = rearrange(
            out, "b h (w1 w2) d -> b w1 w2 (h d)", w1=window_height, w2=window_width
        )
        # combine heads out
        out = self.to_out(out)

        return rearrange(out, "(b x y) ... -> b x y ...", x=height, y=width), attn


class SelfAttentionB(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0.0):
        super().__init__()
        assert (dim % dim_head) == 0, (
            "dimension should be divisible by dimension per head"
        )
        self.heads = dim // dim_head
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(self, x, attn):
        _batch, height, width, window_height, window_width, _, _device, h = (
            *x.shape,
            x.device,
            self.heads,
        )
        # flatten
        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")
        # project for queries, keys, values
        v = self.to_v(x)
        # split heads
        v = rearrange(v, "b n (h d ) -> b h n d", h=h)
        # aggregate
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        # merge heads
        out = rearrange(
            out, "b h (w1 w2) d -> b w1 w2 (h d)", w1=window_height, w2=window_width
        )
        # combine heads out
        out = self.to_out(out)

        return rearrange(out, "(b x y) ... -> b x y ...", x=height, y=width)


class Channel_Attention(nn.Module):
    def __init__(self, dim, heads, bias=False, window_size=7, flash_attn=True):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.ps = window_size
        self.flash_attn = flash_attn
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _b, _c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)
        q, k, v = (
            rearrange(
                t,
                "b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)",
                ph=self.ps,
                pw=self.ps,
                head=self.heads,
            )
            for t in qkv
        )
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        if self.flash_attn:
            with torch.no_grad():
                out = nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out = attn @ v

        out = rearrange(
            out,
            "b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)",
            h=h // self.ps,
            w=w // self.ps,
            ph=self.ps,
            pw=self.ps,
            head=self.heads,
        )

        return self.project_out(out)


class Channel_Attention_grid(nn.Module):
    def __init__(self, dim, heads, bias=False, window_size=7, flash_attn=True):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.ps = window_size
        self.flash_attn = flash_attn
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _b, _c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = (
            rearrange(
                t,
                "b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)",
                ph=self.ps,
                pw=self.ps,
                head=self.heads,
            )
            for t in qkv
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        if self.flash_attn:
            with torch.no_grad():
                out = nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out = attn @ v

        out = rearrange(
            out,
            "b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)",
            h=h // self.ps,
            w=w // self.ps,
            ph=self.ps,
            pw=self.ps,
            head=self.heads,
        )

        return self.project_out(out)


class SLA_A(nn.Module):
    def __init__(self, channel_num=64, window_size=8, with_pe=False, dropout=0.0):
        super().__init__()

        self.w = window_size
        self.norm = nn.LayerNorm(channel_num)  # prenormresidual
        self.attn = SelfAttentionA(
            dim=channel_num,
            dim_head=channel_num,
            dropout=dropout,
            window_size=self.w,
            with_pe=with_pe,
        )
        self.cnorm = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num)

    def forward(self, x):
        x_ = rearrange(x, "b d (x w1) (y w2) -> b x y w1 w2 d", w1=self.w, w2=self.w)
        x, a = self.attn(self.norm(x_))
        x = rearrange(x + x_, "b x y w1 w2 d -> b d (x w1) (y w2)")
        x = self.gfn(self.cnorm(x)) + x
        return x, a


class SGA_A(nn.Module):
    def __init__(self, channel_num=64, window_size=8, with_pe=False, dropout=0.0):
        super().__init__()

        self.w = window_size
        self.norm = nn.LayerNorm(channel_num)  # prenormresidual
        self.attn = SelfAttentionA(
            dim=channel_num,
            dim_head=channel_num,
            dropout=dropout,
            window_size=self.w,
            with_pe=with_pe,
        )
        self.cnorm = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num)

    def forward(self, x):
        x_ = rearrange(x, "b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=self.w, w2=self.w)
        x, a = self.attn(self.norm(x_))
        x = rearrange(x + x_, "b x y w1 w2 d -> b d (w1 x) (w2 y)")
        x = self.gfn(self.cnorm(x)) + x
        return x, a


class SLA_B(nn.Module):
    def __init__(self, channel_num=64, window_size=8, dropout=0.0):
        super().__init__()

        self.w = window_size
        self.norm = nn.LayerNorm(channel_num)  # prenormresidual
        self.attn = SelfAttentionB(
            dim=channel_num, dim_head=channel_num, dropout=dropout
        )
        self.cnorm = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num)

    def forward(self, x, a):
        x_ = rearrange(x, "b d (x w1) (y w2) -> b x y w1 w2 d", w1=self.w, w2=self.w)
        x = self.attn(self.norm(x_), a)
        x = rearrange(x + x_, "b x y w1 w2 d -> b d (x w1) (y w2)")
        return self.gfn(self.cnorm(x)) + x


class SGA_B(nn.Module):
    def __init__(self, channel_num=64, window_size=8, dropout=0.0):
        super().__init__()

        self.w = window_size
        self.norm = nn.LayerNorm(channel_num)  # prenormresidual
        self.attn = SelfAttentionB(
            dim=channel_num, dim_head=channel_num, dropout=dropout
        )
        self.cnorm = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num)

    def forward(self, x, a):
        x_ = rearrange(x, "b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=self.w, w2=self.w)
        x = self.attn(self.norm(x_), a)
        x = rearrange(x + x_, "b x y w1 w2 d -> b d (w1 x) (w2 y)")
        return self.gfn(self.cnorm(x)) + x


class TrBlockC(nn.Module):
    def __init__(self, channel_num=64, window_size=8, flash_attn=True):
        super().__init__()

        self.canorm1 = LayerNorm2d(channel_num)
        self.canorm2 = LayerNorm2d(channel_num)
        self.cttn1 = Channel_Attention(
            dim=channel_num, heads=1, window_size=window_size, flash_attn=flash_attn
        )
        self.cttn2 = Channel_Attention_grid(
            dim=channel_num, heads=1, window_size=window_size, flash_attn=flash_attn
        )

        self.cnorm1 = LayerNorm2d(channel_num)
        self.cnorm2 = LayerNorm2d(channel_num)

        self.gfn1 = Gated_Conv_FeedForward(dim=channel_num)
        self.gfn2 = Gated_Conv_FeedForward(dim=channel_num)

    def forward(self, x):
        x = self.cttn1(self.canorm1(x)) + x
        x = self.gfn1(self.cnorm1(x)) + x
        x = self.cttn2(self.canorm2(x)) + x
        return self.gfn2(self.cnorm2(x)) + x


class IDSA_Block1(nn.Module):
    def __init__(
        self, channel_num=64, window_size=8, with_pe=False, dropout=0.0, flash_attn=True
    ):
        super().__init__()

        self.ch_split = channel_num // 4
        self.LB1 = LB(
            channel_num,
            channel_num,
            downsample=False,
            expansion_rate=1,
            shrinkage_rate=0.25,
        )
        self.LB2 = LB(
            channel_num // 4 * 3,
            channel_num // 4 * 3,
            downsample=False,
            expansion_rate=1,
            shrinkage_rate=0.25,
        )
        self.LB3 = LB(
            channel_num // 2,
            channel_num // 2,
            downsample=False,
            expansion_rate=1,
            shrinkage_rate=0.25,
        )

        self.SLA1 = SLA_A(channel_num // 4 * 3, window_size, with_pe, dropout)
        self.SLA2 = SLA_A(channel_num // 2, window_size, with_pe, dropout)
        self.SLA3 = SLA_A(channel_num // 4, window_size, with_pe, dropout)

        self.SGA1 = SGA_A(channel_num // 4 * 3, window_size, with_pe, dropout)
        self.SGA2 = SGA_A(channel_num // 2, window_size, with_pe, dropout)
        self.SGA3 = SGA_A(channel_num // 4, window_size, with_pe, dropout)

        self.BlockC = TrBlockC(channel_num, window_size, flash_attn)

        self.pw1 = nn.Conv2d(channel_num, channel_num, 1)
        self.pw2 = nn.Conv2d(channel_num // 4 * 3, channel_num // 4 * 3, 1)
        self.pw3 = nn.Conv2d(channel_num, channel_num, 1)

    def forward(self, x):
        x = self.LB1(x)  # 4c->4c
        loc = x[:, : self.ch_split, :, :]  # c
        x, a1 = self.SLA1(x[:, self.ch_split :, :, :])  # 3c->3c
        x, a2 = self.SGA1(x)
        x = self.pw1(torch.cat((x, loc), 1))  # 4c

        x1 = x[:, : self.ch_split, :, :]  # c
        x = self.LB2(x[:, self.ch_split :, :, :])  # 3c
        loc = x[:, : self.ch_split, :, :]  # c
        x, a3 = self.SLA2(x[:, self.ch_split :, :, :])  # 2c->2c
        x, a4 = self.SGA2(x)
        x = self.pw2(torch.cat((x, loc), 1))  # 3c

        x2 = x[:, : self.ch_split, :, :]  # c
        x = self.LB3(x[:, self.ch_split :, :, :])  # 2c
        loc = x[:, : self.ch_split, :, :]  # c
        x, a5 = self.SLA3(x[:, self.ch_split :, :, :])  # c->c
        x, a6 = self.SGA3(x)
        x = self.pw3(torch.cat((x1, x2, x, loc), 1))  # 4c

        x = self.BlockC(x)
        return x, a1, a2, a3, a4, a5, a6


class IDSA_Block2(nn.Module):
    def __init__(self, channel_num=64, window_size=8, dropout=0.0, flash_attn=True):
        super().__init__()

        self.ch_split = channel_num // 4
        self.LB1 = LB(
            channel_num,
            channel_num,
            downsample=False,
            expansion_rate=1,
            shrinkage_rate=0.25,
        )
        self.LB2 = LB(
            channel_num // 4 * 3,
            channel_num // 4 * 3,
            downsample=False,
            expansion_rate=1,
            shrinkage_rate=0.25,
        )
        self.LB3 = LB(
            channel_num // 2,
            channel_num // 2,
            downsample=False,
            expansion_rate=1,
            shrinkage_rate=0.25,
        )

        self.SLA1 = SLA_B(channel_num // 4 * 3, window_size, dropout)
        self.SLA2 = SLA_B(channel_num // 2, window_size, dropout)
        self.SLA3 = SLA_B(channel_num // 4, window_size, dropout)

        self.SGA1 = SGA_B(channel_num // 4 * 3, window_size, dropout)
        self.SGA2 = SGA_B(channel_num // 2, window_size, dropout)
        self.SGA3 = SGA_B(channel_num // 4, window_size, dropout)

        self.BlockC = TrBlockC(channel_num, window_size, flash_attn)

        self.pw1 = nn.Conv2d(channel_num, channel_num, 1)
        self.pw2 = nn.Conv2d(channel_num // 4 * 3, channel_num // 4 * 3, 1)
        self.pw3 = nn.Conv2d(channel_num, channel_num, 1)

    def forward(self, x, a1, a2, a3, a4, a5, a6):
        x = self.LB1(x)  # 4c->4c
        loc = x[:, : self.ch_split, :, :]  # c
        x = self.SLA1(x[:, self.ch_split :, :, :], a1)  # 3c->3c
        x = self.SGA1(x, a2)
        x = self.pw1(torch.cat((x, loc), 1))  # 4c

        x1 = x[:, : self.ch_split, :, :]  # c
        x = self.LB2(x[:, self.ch_split :, :, :])  # 3c
        loc = x[:, : self.ch_split, :, :]  # c
        x = self.SLA2(x[:, self.ch_split :, :, :], a3)  # 2c->2c
        x = self.SGA2(x, a4)
        x = self.pw2(torch.cat((x, loc), 1))  # 3c

        x2 = x[:, : self.ch_split, :, :]  # c
        x = self.LB3(x[:, self.ch_split :, :, :])  # 2c
        loc = x[:, : self.ch_split, :, :]  # c
        x = self.SLA3(x[:, self.ch_split :, :, :], a5)  # c->c
        x = self.SGA3(x, a6)
        x = self.pw3(torch.cat((x1, x2, x, loc), 1))  # 4c

        return self.BlockC(x)


class IDSG(nn.Module):
    def __init__(self, channel_num=64, window_size=8, bias=True, flash_attn=True):
        super().__init__()

        self.residual_layer = IDSA_Block2(
            channel_num, window_size=window_size, flash_attn=flash_attn
        )
        self.res_end = nn.Conv2d(channel_num, channel_num, 1, 1, 0, bias=bias)
        esa_channel = max(channel_num // 4, 16)
        self.esa = ESA(esa_channel, channel_num)

    def forward(self, x, a1, a2, a3, a4, a5, a6):
        out = self.residual_layer(x, a1, a2, a3, a4, a5, a6)
        out = self.res_end(out)
        out = out + x
        return self.esa(out)


class IDSG_A(nn.Module):
    def __init__(
        self,
        channel_num=64,
        window_size=8,
        block_num=4,
        pe=False,
        bias=True,
        flash_attn=True,
    ):
        super().__init__()

        group_list = []
        for _ in range(block_num):
            temp_res = IDSA_Block1(
                channel_num, window_size=window_size, with_pe=pe, flash_attn=flash_attn
            )
            group_list.append(temp_res)
        self.res_end = nn.Conv2d(channel_num, channel_num, 1, 1, 0, bias=bias)
        self.residual_layer = nn.Sequential(*group_list)
        esa_channel = max(channel_num // 4, 16)
        self.esa = ESA(esa_channel, channel_num)

    def forward(self, x):
        out, a1, a2, a3, a4, a5, a6 = self.residual_layer(x)
        out = self.res_end(out)
        out = out + x
        return self.esa(out), a1, a2, a3, a4, a5, a6


@ARCH_REGISTRY.register()
class asid(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        res_num=3,
        block_num=1,
        window_size=8,
        scale=upscale,
        flash_attn=True,
        d8=False,
        pe=False,
        bias=True,
        drop=0.0,
    ):
        super().__init__()

        self.res_num = res_num
        self.window_size = window_size
        self.up_scale = scale
        self.d8 = d8
        self.drop = nn.Dropout2d(drop)

        self.block0 = IDSG_A(
            channel_num=num_feat,
            block_num=block_num,
            window_size=window_size,
            pe=pe,
            bias=bias,
            flash_attn=flash_attn,
        )
        self.block1 = IDSG(
            channel_num=num_feat,
            window_size=window_size,
            bias=bias,
            flash_attn=flash_attn,
        )
        self.block2 = IDSG(
            channel_num=num_feat,
            window_size=window_size,
            bias=bias,
            flash_attn=flash_attn,
        )
        if self.d8:
            self.block3 = IDSG(
                channel_num=num_feat,
                window_size=window_size,
                bias=bias,
                flash_attn=flash_attn,
            )
            self.block4 = IDSG(
                channel_num=num_feat,
                window_size=window_size,
                bias=bias,
                flash_attn=flash_attn,
            )
            self.block5 = IDSG(
                channel_num=num_feat,
                window_size=window_size,
                bias=bias,
                flash_attn=flash_attn,
            )
            self.block6 = IDSG(
                channel_num=num_feat,
                window_size=window_size,
                bias=bias,
                flash_attn=flash_attn,
            )
            self.block7 = IDSG(
                channel_num=num_feat,
                window_size=window_size,
                bias=bias,
                flash_attn=flash_attn,
            )

        self.input = nn.Conv2d(
            in_channels=num_in_ch,
            out_channels=num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.output = nn.Conv2d(
            in_channels=num_feat,
            out_channels=num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.up = pixelshuffle_block(num_feat, num_out_ch, self.up_scale, bias=bias)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant", 0)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual = self.input(x)
        out, a1, a2, a3, a4, a5, a6 = self.block0(residual)
        out = self.block1(out, a1, a2, a3, a4, a5, a6)
        out = self.block2(out, a1, a2, a3, a4, a5, a6)
        if self.d8:
            out = self.block3(out, a1, a2, a3, a4, a5, a6)
            out = self.block4(out, a1, a2, a3, a4, a5, a6)
            out = self.block5(out, a1, a2, a3, a4, a5, a6)
            out = self.block6(out, a1, a2, a3, a4, a5, a6)
            out = self.block7(out, a1, a2, a3, a4, a5, a6)

        # origin
        if self.training:
            out = self.drop(out)
        out = torch.add(self.output(out), residual)
        out = self.up(out)
        return out[:, :, : H * self.up_scale, : W * self.up_scale]


@ARCH_REGISTRY.register()
def asid_d8(**kwargs):
    return asid(res_num=8, d8=True, **kwargs)
