from typing import Literal
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from .conv2d_resample import conv2d_resample
from .upfirdn2d import setup_filter, upsample2d
from .bias_act import activation_funcs, bias_act



def normalize_2nd_moment(x: Tensor, dim: int = 1) -> Tensor:
    return (
        x * (
            x.square().mean(dim=dim, keepdim=True)
            + torch.finfo(x.dtype).eps
        ).rsqrt()
    )



class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features: int,                # Number of input features.
        out_features: int,               # Number of output features.
        bias: bool = True,     # Apply additive bias before the activation function?
        activation: str = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: int = 1,        # Learning rate multiplier.
        bias_init: int | float = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier
        )
        self.bias = (
            torch.nn.Parameter(
                torch.full([out_features], bias_init, dtype=torch.float32)
            )
            if bias
            else None
        )
        self.activation = activation

        self.weight_gain: float = lr_multiplier / np.sqrt(in_features)
        self.bias_gain: int = lr_multiplier


    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.device, dtype=x.dtype) * self.weight_gain
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        if b is not None and self.bias_gain != 1:
            b = b.to(x.device) * self.bias_gain

        if self.activation == 'linear' and b is not None:
            # out = torch.addmm(b.unsqueeze(0), x, w.t())
            x = x.matmul(w.t())
            out = x + b.reshape(
                [-1 if i == x.ndim-1 else 1 for i in range(x.ndim)]
            ).to(x.device)
        else:
            x = x.matmul(w.t())
            out = bias_act(x, b, act=self.activation, dim=x.ndim-1)
        return out



class Conv2dLayer(nn.Module):
    def __init__(self,
        in_channels: int,           # Number of input channels.
        out_channels: int,          # Number of output channels.
        kernel_size: int,           # Width and height of the convolution kernel.
        bias: bool = True,          # Apply additive bias before the activation function?
        activation: str = 'linear', # Activation function: 'relu', 'lrelu', etc.
        up: int= 1,                 # Integer upsampling factor.
        down: int= 1,               # Integer downsampling factor.
        resample_filter: list[int] = [1, 3, 3, 1], # Low-pass filter to apply when resampling activations.
        conv_clamp: bool= None,     # Clamp the output to +-X, None = disable clamping.
        trainable: bool= True,      # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.register_buffer('resample_filter', setup_filter(resample_filter))
        self.conv_clamp = conv_clamp
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = activation_funcs[activation].def_gain

        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None


    def forward(self, x: Tensor, gain: int = 1) -> Tensor:
        w: Tensor = self.weight.to(dtype=x.dtype) * self.weight_gain
        x = conv2d_resample(
            x=x,
            w=w,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        out = bias_act(
            x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp
        )
        return out



class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,   # Number of input channels.
        out_channels: int,  # Number of output channels.
        kernel_size: int,   # Width and height of the convolution kernel.
        style_dim: int,     # dimension of the style code
        demodulate=True,    # perfrom demodulation
        up: int = 1,        # Integer upsampling factor.
        down: int = 1,      # Integer downsampling factor.
        resample_filter: list[int] = [1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,    # Clamp the output to +-X, None = disable clamping.
    ):
        super().__init__()
        self.demodulate = demodulate

        self.weight = torch.nn.Parameter(
            torch.randn([1, out_channels, in_channels, kernel_size, kernel_size])
        )
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.padding = self.kernel_size // 2
        self.up = up
        self.down = down
        self.register_buffer('resample_filter', setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

        self.affine = FullyConnectedLayer(style_dim, in_channels, bias_init=1)

    def forward(self, x: Tensor, style) -> Tensor:
        batch, in_channels, height, width = x.shape
        style = self.affine(style).view(batch, 1, in_channels, 1, 1)
        weight = self.weight.to(dtype=x.dtype) * self.weight_gain * style

        if self.demodulate:
            decoefs = (weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
            weight = weight * decoefs.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels,
            in_channels,
            self.kernel_size,
            self.kernel_size
        )
        x = x.view(1, batch * in_channels, height, width)
        x = conv2d_resample(
            x=x,
            w=weight,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            groups=batch
        )
        out = x.view(batch, self.out_channels, *x.shape[2:])

        return out



class StyleConv(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        style_dim,                      # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = False,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        demodulate      = True,         # perform demodulation
    ):
        super().__init__()

        self.conv = ModulatedConv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    style_dim=style_dim,
                                    demodulate=demodulate,
                                    up=up,
                                    resample_filter=resample_filter,
                                    conv_clamp=conv_clamp)

        self.use_noise = use_noise
        self.resolution = resolution
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.activation = activation
        self.act_gain = activation_funcs[activation].def_gain
        self.conv_clamp = conv_clamp

    def forward(
        self,
        x,
        style,
        noise_mode: Literal['random', 'const', 'none'] = 'random',
        gain=1
    ):
        x = self.conv(x, style)
        if self.use_noise:
            if noise_mode == 'random':
                xh, xw = x.size()[-2:]
                noise = torch.randn([x.shape[0], 1, xh, xw], device=x.device) \
                        * self.noise_strength
            if noise_mode == 'const':
                noise = self.noise_const * self.noise_strength
            x = x + noise

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        out = bias_act(x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp)

        return out



class ToRGB(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim,
        kernel_size: int = 1,
        resample_filter: list[int, int, int, int] = [1,3,3,1],
        conv_clamp = None,
        demodulate: bool = False
    ):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            demodulate=demodulate,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.register_buffer('resample_filter', setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

    def forward(self, x: Tensor, style, skip: Tensor | None = None) -> Tensor:
        x = self.conv(x, style)
        out: Tensor = bias_act(x, self.bias, clamp=self.conv_clamp)

        if skip is not None:
            if skip.shape != out.shape:
                skip = upsample2d(skip, self.resample_filter)
            out = out + skip

        return out



def get_style_code(a, b):
    return torch.cat([a, b], dim=1)



class DecBlockFirst(nn.Module):
    def __init__(self, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.fc = FullyConnectedLayer(in_features=in_channels*2,
                                      out_features=in_channels*4**2,
                                      activation=activation)
        self.conv = StyleConv(in_channels=in_channels,
                              out_channels=out_channels,
                              style_dim=style_dim,
                              resolution=4,
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

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = x + E_features[2]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img



class DecBlockFirstV2(nn.Module):
    def __init__(self, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
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
            resolution=4,
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

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + E_features[2]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img



class DecBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):  # res = 2, ..., resolution_log2
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            up=2,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.conv1 = StyleConv(
            in_channels=out_channels,
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


    def forward(self, x, img, ws, gs, E_features, noise_mode='random'):
        style = get_style_code(ws[:, self.res * 2 - 5], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 4], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 3], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img



class MappingNet(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers: int= 8,         # Number of mapping layers.
        embed_features = None,      # Label embedding dimensionality, None = same as w_dim.
        layer_features = None,      # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation: str = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 0.01,    # Learning rate multiplier for the mapping layers.
        w_avg_beta: float= 0.995,   # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = (
            [z_dim + embed_features]
            + [layer_features] * (num_layers - 1)
            + [w_dim]
        )

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation=activation,
                lr_multiplier=lr_multiplier
            )
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))


    def forward(
        self,
        z: Tensor,
        c,
        truncation_psi = 1,
        truncation_cutoff = None,
        skip_w_avg_update: bool = False
    ):
        # Embed, normalize, and concat inputs.
        x = None
        if self.z_dim > 0:
            x = normalize_2nd_moment(z)
        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embed(c))
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f"fc{idx}")
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            # assert self.w_avg_beta is not None
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = (
                    self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
                )

        return x
