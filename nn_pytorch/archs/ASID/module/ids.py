import torch
import torch.nn as nn
from torch import Tensor, einsum
import  torch.nn.functional as F

from .esa import ESA
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from .layernorm import LayerNorm2d


class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 1, bias=False):
        super().__init__()
        hidden_features = int(dim * mult)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1")
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate(x)



class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x



class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob

    def forward(self, x: Tensor) -> Tensor:
        if self.prob == 0. or not self.training:
            return x

        ft: Tensor = torch.FloatTensor((x.shape[0], 1, 1, 1), device=x.device).uniform_()
        keep_mask = ft > self.prob
        return x * keep_mask / (1 - self.prob)



def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate=0.25,
    dropout = 0.
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
        nn.Conv2d(hidden_dim, dim_out, 1)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net



class LB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """
    def __init__(
        self,
        dim_in,
        dim_out,
        downsample = False,
        expansion_rate = 1,
        shrinkage_rate=0.25
    ):
        super().__init__()
        self.net = MBConv(
            dim_in,
            dim_out,
            downsample=downsample,
            expansion_rate=expansion_rate,
            shrinkage_rate=shrinkage_rate
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)



class SelfAttentionA(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7,
        with_pe = True,
    ):
        super().__init__()
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.with_pe = with_pe
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos))
            grid = rearrange(grid, "c i j -> (i j) c")
            rel_pos = (
                rearrange(grid, "i ... -> i 1 ...") - rearrange(grid, "j ... -> 1 j ...")
            )
            rel_pos += window_size - 1
            rel_pos_indices = (
                rel_pos * torch.tensor([2 * window_size - 1, 1])
            ).sum(dim=-1)
            self.register_buffer(
                "rel_pos_indices", rel_pos_indices, persistent=False
            )


    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, window_height, window_width, _, device, h = (
            *x.shape, x.device, self.heads
        )

        # flatten
        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d ) -> b h n d", h=h), (q, k, v)
        )

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
    def __init__(
        self,
        dim: int,
        dim_head: int = 32,
        dropout: float = 0.,
    ):
        super().__init__()
        self.heads = dim // dim_head
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )


    def forward(self, x: Tensor, attn) -> Tensor:
        batch, height, width, window_height, window_width, _, device, h = (
            *x.shape, x.device, self.heads
        )

        # flatten
        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")

        # project for queries, keys, values
        v = self.to_v(x)

        # split heads
        v = rearrange(v, "b n (h d ) -> b h n d", h = h)

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
    def __init__(
        self,
        dim,
        heads,
        bias=False,
        window_size = 7
    ):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.ps = window_size
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)",
                ph=self.ps, pw=self.ps, head=self.heads
            ),
            qkv
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out =  (attn @ v)

        out = rearrange(
            out,
            "b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)",
            h=h//self.ps,
            w=w//self.ps,
            ph=self.ps,
            pw=self.ps,
            head=self.heads
        )
        return self.project_out(out)



class Channel_Attention_grid(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        bias=False,
        window_size = 7
    ):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.ps = window_size
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)",
                ph=self.ps,
                pw=self.ps,
                head=self.heads
            ),
            qkv
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out =  (attn @ v)

        out = rearrange(
            out,
            "b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)",
            h=h // self.ps,
            w=w // self.ps,
            ph=self.ps,
            pw=self.ps,
            head=self.heads
        )
        return self.project_out(out)



class SLA_A(nn.Module):
    def __init__(self,
        channel_num: int = 64,
        window_size: int = 8,
        with_pe: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.w = window_size
        self.norm = nn.LayerNorm(channel_num) # prenormresidual
        self.attn = SelfAttentionA(
            dim=channel_num,
            dim_head=channel_num,
            dropout=dropout,
            window_size=window_size,
            with_pe=with_pe
        )
        self.cnorm = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num)


    def forward(self, x: Tensor) -> Tensor:
        x_ = rearrange(
            x, "b d (x w1) (y w2) -> b x y w1 w2 d", w1=self.w, w2=self.w
        )
        x, a = self.attn(self.norm(x_))
        x = rearrange(x + x_, "b x y w1 w2 d -> b d (x w1) (y w2)")
        x = self.gfn(self.cnorm(x)) + x
        return x, a



class SGA_A(nn.Module):
    def __init__(
        self,
        channel_num=64,
        window_size=8,
        with_pe=False,
        dropout=0.0
    ):
        super().__init__()
        self.w = window_size
        self.norm = nn.LayerNorm(channel_num) # prenormresidual
        self.attn = SelfAttentionA(
            dim=channel_num,
            dim_head=channel_num,
            dropout=dropout,
            window_size=self.w,
            with_pe=with_pe
        )
        self.cnorm = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num)


    def forward(self, x: Tensor) -> Tensor:
        x_ = rearrange(
            x, "b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=self.w, w2=self.w
        )
        x, a = self.attn(self.norm(x_))
        x = rearrange(x + x_, "b x y w1 w2 d -> b d (w1 x) (w2 y)")
        x = self.gfn(self.cnorm(x)) + x
        return x, a



class SLA_B(nn.Module):
    def __init__(self,
        channel_num=64,
        window_size=8,
        dropout=0.0
    ):
        super().__init__()
        self.w = window_size
        # prenormresidual
        self.norm = nn.LayerNorm(channel_num)
        self.attn = SelfAttentionB(
            dim=channel_num, dim_head=channel_num, dropout=dropout,
        )
        self.cnorm = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num)


    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        x_ = rearrange(
            x, "b d (x w1) (y w2) -> b x y w1 w2 d", w1=self.w, w2=self.w
        )
        x = self.attn(self.norm(x_), a)
        x = rearrange(x + x_, "b x y w1 w2 d -> b d (x w1) (y w2)")
        x = self.gfn(self.cnorm(x)) + x
        return x



class SGA_B(nn.Module):
    def __init__(
        self,
        channel_num = 64,
        window_size = 8,
        dropout = 0.0
    ):
        super().__init__()
        self.w = window_size
        self.norm = nn.LayerNorm(channel_num)
        self.attn = SelfAttentionB(
            dim=channel_num,
            dim_head=channel_num,
            dropout=dropout,
        )
        self.cnorm = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num)


    def forward(self, x: Tensor, a: Tensor) -> Tensor:
        x_ = rearrange(
            x, "b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=self.w, w2=self.w
        )
        x = self.attn(self.norm(x_),a)
        x = rearrange(x + x_, "b x y w1 w2 d -> b d (w1 x) (w2 y)")
        x = self.gfn(self.cnorm(x)) + x
        return x



class TrBlockC(nn.Module):
    def __init__(self, channel_num=64, window_size=8):
        super().__init__()

        self.canorm1 = LayerNorm2d(channel_num)
        self.canorm2 = LayerNorm2d(channel_num)
        self.cttn1 = Channel_Attention(
            dim=channel_num, heads=1, window_size=window_size
        )
        self.cttn2 = Channel_Attention_grid(
            dim=channel_num, heads=1, window_size=window_size
        )

        self.cnorm1 = LayerNorm2d(channel_num)
        self.cnorm2 = LayerNorm2d(channel_num)

        self.gfn1 = Gated_Conv_FeedForward(dim=channel_num)
        self.gfn2 = Gated_Conv_FeedForward(dim=channel_num)


    def forward(self, x: Tensor) -> Tensor:
        x = self.cttn1(self.canorm1(x)) + x
        x = self.gfn1(self.cnorm1(x)) + x
        x = self.cttn2(self.canorm2(x)) + x
        x = self.gfn2(self.cnorm2(x)) + x
        return x



class IDSA_Block1(nn.Module):
    def __init__(
        self,
        channel_num: int = 64,
        window_size: int = 8,
        with_pe: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.w= 8
        self.ch_split = channel_num//4

        self.LB1 = LB(
            dim_in=channel_num,
            dim_out=channel_num,
            downsample=False,
            expansion_rate=1,
            shrinkage_rate=0.25
        )
        self.LB2 = LB(
            dim_in=channel_num // 4*3,
            dim_out=channel_num // 4 * 3,
            downsample=False,
            expansion_rate=1,
            shrinkage_rate=0.25
        )
        self.LB3 = LB(
            dim_in=channel_num // 2,
            dim_out=channel_num//2,
            downsample=False,
            expansion_rate=1,
            shrinkage_rate=0.25
        )

        self.SLA1 = SLA_A(channel_num // 4 * 3, window_size=window_size, with_pe=with_pe, dropout=dropout)
        self.SLA2 = SLA_A(channel_num // 2, window_size=window_size, with_pe=with_pe, dropout=dropout)
        self.SLA3 = SLA_A(channel_num // 4, window_size=window_size, with_pe=with_pe, dropout=dropout)

        self.SGA1 = SGA_A(channel_num // 4 * 3, window_size=window_size, with_pe=with_pe, dropout=dropout)
        self.SGA2 = SGA_A(channel_num // 2, window_size=window_size, with_pe=with_pe, dropout=dropout)
        self.SGA3 = SGA_A(channel_num // 4, window_size=window_size, with_pe=with_pe, dropout=dropout)

        self.BlockC = TrBlockC(channel_num, window_size=window_size)

        self.pw1 = nn.Conv2d(channel_num, channel_num, 1)
        self.pw2 = nn.Conv2d(channel_num // 4 * 3, channel_num // 4 * 3, 1)
        self.pw3 = nn.Conv2d(channel_num, channel_num, 1)


    def forward(self, x: Tensor) -> Tensor:
        x = self.LB1(x) #4c->4c
        loc = x[:, :self.ch_split, :, :] #c
        x, a1 = self.SLA1(x[:, self.ch_split:, :, :]) #3c->3c
        x, a2 = self.SGA1(x)

        x = self.pw1(torch.cat((x, loc), 1)) # 4c

        x1 = x[:, :self.ch_split, :, :] #c

        x = self.LB2(x[:,self.ch_split:, :, :]) #3c
        loc = x[:, :self.ch_split, :, :] #c
        x, a3 = self.SLA2(x[:, self.ch_split:, :, :]) #2c->2c
        x, a4 = self.SGA2(x)

        x = self.pw2(torch.cat((x, loc), 1)) # 3c
        x2 = x[:, :self.ch_split, :, :] #c

        x = self.LB3(x[:, self.ch_split:, :, :]) #2c
        loc = x[:, :self.ch_split, :, :] #c
        x, a5 = self.SLA3(x[:, self.ch_split:, :, :]) #c->c
        x, a6 = self.SGA3(x)

        x = self.pw3(torch.cat((x1, x2, x, loc), 1)) # 4c

        x = self.BlockC(x)
        return x, a1, a2, a3, a4, a5, a6



class IDSA_Block2(nn.Module):
    def __init__(
        self,
        channel_num: int = 64,
        window_size: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.w = window_size
        self.ch_split = channel_num // 4

        self.LB1 = LB(
            dim_in=channel_num,
            dim_out=channel_num,
            downsample=False,
            expansion_rate=1,
            shrinkage_rate=0.25
        )
        self.LB2 = LB(
            dim_in=channel_num // 4 * 3,
            dim_out=channel_num // 4 * 3,
            downsample=False,
            expansion_rate=1,
            shrinkage_rate=0.25
        )
        self.LB3 = LB(
            dim_in=channel_num // 2,
            dim_out=channel_num // 2,
            downsample=False,
            expansion_rate=1,
            shrinkage_rate=0.25
        )

        self.SLA1 = SLA_B(channel_num // 4 * 3, window_size=window_size, dropout=dropout)
        self.SLA2 = SLA_B(channel_num // 2, window_size=window_size, dropout=dropout)
        self.SLA3 = SLA_B(channel_num // 4, window_size=window_size, dropout=dropout)

        self.SGA1 = SGA_B(channel_num // 4 * 3, window_size=window_size, dropout=dropout)
        self.SGA2 = SGA_B(channel_num // 2, window_size=window_size, dropout=dropout)
        self.SGA3 = SGA_B( channel_num // 4, window_size=window_size, dropout=dropout)

        self.BlockC = TrBlockC(channel_num, window_size)
        self.pw1 = nn.Conv2d(channel_num, channel_num, 1)
        self.pw2 = nn.Conv2d(channel_num // 4 * 3, channel_num // 4 * 3, 1)
        self.pw3 = nn.Conv2d(channel_num, channel_num, 1)


    def forward(self, x: Tensor, a1, a2, a3, a4, a5, a6) -> Tensor:
        x = self.LB1(x) #4c->4c
        loc = x[:, :self.ch_split, :, :] #c
        x = self.SLA1(x[:, self.ch_split:, :, :],a1) #3c->3c
        x = self.SGA1(x,a2)

        x = self.pw1(torch.cat((x,loc),1)) # 4c

        x1 = x[:, :self.ch_split, :, :] #c

        x = self.LB2(x[:, self.ch_split:, :, :]) #3c
        loc = x[:,:self.ch_split,:,:] #c
        x = self.SLA2(x[:, self.ch_split:, :, :], a3) #2c->2c
        x = self.SGA2(x, a4)

        x = self.pw2(torch.cat((x, loc), 1)) # 3c
        x2 = x[:,:self.ch_split,:,:] #c

        x = self.LB3(x[:, self.ch_split:, :, :]) #2c
        loc = x[:, :self.ch_split, :, :] #c
        x = self.SLA3(x[:, self.ch_split:, :, :], a5) #c->c
        x = self.SGA3(x, a6)

        x = self.pw3(torch.cat((x1, x2, x, loc), 1)) # 4c
        return self.BlockC(x)



class IDSG(nn.Module):
    def __init__(
        self,
        channel_num: int = 64,
        bias: bool = True,
        window_size: int = 8,
        pe: bool = False,
    ):
        super().__init__()

        self.residual_layer = IDSA_Block2(
            channel_num, window_size=window_size,
        )
        self.res_end = nn.Conv2d(channel_num, channel_num, 1, 1, 0, bias=bias)
        esa_channel = max(channel_num // 4, 16)
        self.esa = ESA(esa_channel, channel_num)


    def forward(self, x: Tensor, a1, a2, a3, a4, a5, a6) -> Tensor:
        out = self.residual_layer(x, a1, a2, a3, a4, a5, a6)
        out = self.res_end(out)
        out = out + x
        return self.esa(out)



class IDSG_A(nn.Module):
    def __init__(
        self,
        channel_num: int = 64,
        bias: bool = True,
        block_num: int = 4,
        window_size: int = 8,
        pe: bool = False,
    ):
        super().__init__()

        group_list = []
        for _ in range(block_num):
            temp_res = IDSA_Block1(
                channel_num, window_size=window_size, with_pe=pe
            )
            group_list.append(temp_res)
        self.res_end = nn.Conv2d(channel_num, channel_num, 1, 1, 0, bias=bias)
        self.residual_layer = nn.Sequential(*group_list)
        esa_channel = max(channel_num // 4, 16)
        self.esa = ESA(esa_channel, channel_num)


    def forward(self, x: Tensor) -> Tensor:
        out, a1, a2, a3, a4, a5, a6 = self.residual_layer(x)
        out = self.res_end(out)
        out = out + x
        return self.esa(out), a1, a2, a3, a4, a5, a6




