from collections import OrderedDict
import time
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# import kornia.contrib
from torch import Tensor

from .filters import (
    fourier_gradients,
    recursive_filter
)

from .polyblur_functions import mild_inverse_rank3, blur_estimation

from ..._shared.basicblock import (
    conv,
    downsample_maxpool,
    downsample_avgpool,
    downsample_strideconv,
    ResBlock,
    sequential,
    upsample_upconv,
    upsample_convtranspose,
    upsample_upconv,
    upsample_pixelshuffle,
)




# class OpticsCorrection(nn.Module):
#     def __init__(
#         self,
#         load_weights: bool = True,
#         model_type: Literal["tiny", "super_tiny"] = "tiny",
#         patch_size: int = 400,
#         overlap_percentage: float = 0.25,
#         ker_size: int = 31,
#         batch_size: int = 20,
#         n_angles: int = 6,
#         n_interpolated_angles: int = 30
#     ):
#         super().__init__()
#         ## Sharpening attributes
#         self.patch_size = patch_size
#         self.overlap_percentage = overlap_percentage
#         self.ker_size = ker_size
#         self.batch_size = batch_size

#         ## Defringing attributes
#         if model_type == "tiny":
#             self.defringer = ResUNet(nc=[16, 32, 64, 64], in_nc=2, out_nc=1)
#         elif model_type == "super_tiny":
#             self.defringer = ResUNet(nc=[16, 16, 32, 32], in_nc=2, out_nc=1)
#         else:
#             self.defringer = ResUNet(nc=[64, 128, 256, 512], in_nc=2, out_nc=1)
#         if load_weights:
#             state_dict_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../checkpoints/" + model_type + "_epoch_1000.pt")
#             self.defringer.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
#         self.defringer.eval()

#         ## Build angle arrays for interpolation in polyblur
#         self.thetas = nn.Parameter(torch.linspace(0, 180, n_angles+1).unsqueeze(0), requires_grad=False)   # (1,n)
#         self.interpolated_thetas = nn.Parameter(torch.arange(0, 180, 180 / n_interpolated_angles).unsqueeze(0), requires_grad=False)   # (1,N)
#         self.freqs = nn.Parameter((torch.arange(0, patch_size) - patch_size//2) / patch_size, requires_grad=False)  # [-0.5,...,0.5)


#     def forward(self, image, c=0.358, sigma_b=0.451, polyblur_iteration=1, alpha=2, b=3, q=0,
#                 do_decomposition=False, do_halo_removal=False, do_edgetaper=False,
#                 sigma_s=200, sigma_r=0.1):
#         assert(image.shape[0] == 1)  # One image at the time for test
#         ## Device on which run the computation -- based on where is the model
#         device = self.thetas.device
#         device_image = image.device
#         print("Inference on device %s:" % device)

#         ## Decide which deblurring method use
#         if device == torch.device("cpu"):
#             deblurring_method = "fft"
#         else:
#             deblurring_method = "direct"
#         print("Deblurring method is: %s" % deblurring_method)

#         start = time.time()
#         ## Make sure dimensions are even
#         image, h_full, w_full = self.make_even_dimensions(image)

#         ## Flip the image to be processed in landscape format
#         if h_full > w_full:
#             image = torch.permute(image, (0, 1, 3, 2))
#             h_full, w_full = image.shape[-2:]
#             do_flip = True
#         else:
#             do_flip = False

#         ## If the image is too large (for instance 24Mpixel), break it into sub-images
#         if h_full > 2300 and w_full > 3400:  # If larger than 12Mpixel, break the image
#             h = h_full // 2
#             w = w_full // 2
#             frames = [image[...,0:h,0:w], image[..., 0:h,w:],
#                       image[...,h:, 0:w], image[..., h:, w:]]
#         else:  # Else, just store the image itself
#             h = h_full
#             w = w_full
#             frames = [image]

#         ## Get precomputed CUDA arrays
#         thetas = self.thetas
#         interpolated_thetas = self.interpolated_thetas
#         freqs = None
#         bs = self.batch_size
#         ps = self.patch_size
#         window = self.build_window(ps, window_type="kaiser")
#         window = window.view(1,1,ps,ps)  # (1,1,pH,pW)
#         window = window.to(device, non_blocking=True)
#         print("Init:             %1.3f" % (time.time() - start))

#         ## Loop over all the sub-images
#         for idx_image, image in enumerate(frames):
#             start = time.time()
#             image = image.to(device, non_blocking=True)

#             ## Get the patches
#             window_size = (ps, ps)
#             original_size = (h, w)
#             stride = (int(ps * (1-self.overlap_percentage)), int(ps * (1-self.overlap_percentage)))
#             padding = kornia.contrib.compute_padding(original_size, window_size)
#             # patches is (1,N,C,H,W)
#             patches = kornia.contrib.extract_tensor_patches(image, window_size, stride, padding).squeeze(0)
#             n_blocks = patches.shape[0]

#             ## Create the array for outputing results
#             window_sum = self.fold(window.repeat(n_blocks, 1, 1, 1).unsqueeze(0),
#                                    original_size, window_size, stride, padding)

#             ## Main loop on patches
#             n_chuncks = int(np.ceil(n_blocks / bs))
#             for n in range(n_chuncks):
#                 ## Work on a subset of patches if they are too many of them
#                 first = n * bs
#                 last = min((n+1) * bs, n_blocks)
#                 patch = patches[first:last]

#                 ## Precompute blurry patches gradients for halo_removal
#                 grad_patch = fourier_gradients(patch, freqs)

#                 ##### Blind deblurring module (Polyblur)
#                 if do_decomposition:
#                     ## Run polyblur and the base image of a base/detail decomposition to not
#                     ## magnify noise and compression artifacts.
#                     start = time.time()
#                     patch_base = recursive_filter(patch, sigma_s=sigma_s, sigma_r=sigma_r, num_iterations=3)
#                     print("Decompostion:     %1.3f" % (time.time() - start))
#                     patch_detail = patch - patch_base
#                 else:
#                     ## Run polyblur on the image if noise/artifacts are deemed small enough.
#                     patch_base = patch
#                 for m in range(polyblur_iteration):
#                     start = time.time()
#                     kernel = blur_estimation(patch_base, c=c, sigma_b=sigma_b, ker_size=self.ker_size, q=q,
#                                              thetas=thetas, interpolated_thetas=interpolated_thetas, freqs=freqs)
#                     print("Estimation %d:     %1.3f" % (m+1, time.time() - start))
#                     start = time.time()
#                     patch_base = mild_inverse_rank3(patch_base, kernel, correlate=True,
#                                                     do_halo_removal=do_halo_removal, do_edgetaper=do_edgetaper,
#                                                     alpha=alpha, b=b, method=deblurring_method, grad_img=grad_patch)  # (b,3,pH,pW)
#                     print("Deblurring %d:     %1.3f" % (m+1, time.time() - start))
#                 if do_decomposition:
#                     patch = patch_detail + patch_base
#                 else:
#                     patch = patch_base
#                 patch = patch.clamp(0, 1)

#                 ##### Defringing module
#                 ## Inference
#                 start = time.time()
#                 with torch.no_grad():
#                     patch[:, 0:1] -= self.defringer(patch[:, 0:2])
#                     patch[:, 2:3] -= self.defringer(torch.cat([patch[:, 2:3], patch[:, 1:2]], dim=1))
#                 print("Defringing:       %1.3f" % (time.time() - start))

#                 ## Replace the subset of patches
#                 patches[first:last] = patch

#             start = time.time()
#             frames[idx_image] = self.fold(patches.unsqueeze(0) * window,
#                                     original_size, window_size, stride, padding)
#             frames[idx_image].div_(window_sum)
#             frames[idx_image].clamp_(0, 1)
#             frames[idx_image] = frames[idx_image].to(device_image)
#             print("Stitching:        %1.3f" % (time.time() - start))

#         start = time.time()
#         ## If we fragmented the image, rebuild the full one
#         if len(frames) > 1:
#             image = torch.cat([torch.cat([frames[0], frames[1]], dim=-1),
#                                torch.cat([frames[2], frames[3]], dim=-1)], dim=-2)
#         else:
#             image = frames[0]
#         del frames
#         print("Concatenate:      %1.3f" % (time.time() - start))

#         ## Undo the flip if the image was in portrait format
#         if do_flip:
#             return torch.permute(image, (0, 1, 3, 2))
#         else:
#             return image


#     def fold(self, x: Tensor, original_size, window_size, stride, padding):
#         b,n,c,h,w = x.shape
#         x = x.view(b,n,c*h*w).permute(0,2,1)
#         output_size = (original_size[0] + padding[0] + padding[1],
#                       original_size[1] + padding[2] + padding[3])
#         image = F.fold(x, output_size=output_size, kernel_size=window_size, stride=stride)
#         image = self.crop_with_old_size(image, original_size)
#         return image


#     def make_even_dimensions(self, image):
#         h, w = image.shape[-2:]
#         if h % 2 == 1:
#             image = image[..., :-1, :]
#             h -= 1
#         if w % 2 == 1:
#             image = image[..., :, :-1]
#             w -= 1
#         return image, h, w


#     @staticmethod
#     def pad_with_new_size(img, new_size, mode="constant"):
#         h, w = img.shape[-2:]
#         new_h, new_w = new_size
#         pad_left = int(np.floor((new_w - w)/2))
#         pad_right = int(np.ceil((new_w - w) / 2))
#         pad_top = int(np.floor((new_h - h)/2))
#         pad_bottom = int(np.ceil((new_h - h) / 2))
#         padding = (pad_left, pad_right, pad_top, pad_bottom)
#         img = F.pad(img, padding, mode=mode)
#         return img

#     @staticmethod
#     def crop_with_old_size(img, old_size):
#         h, w, = img.shape[-2:]
#         old_h, old_w = old_size
#         crop_left = int(np.floor((w - old_w)/2))
#         if crop_left > 0:
#             img = img[..., :, crop_left:]
#         crop_right = int(np.ceil((w - old_w) / 2))
#         if crop_right > 0:
#             img = img[..., :, :-crop_right]
#         crop_top = int(np.floor((h - old_h) / 2))
#         if crop_top > 0:
#             img = img[..., crop_top:, :]
#         crop_bottom = int(np.ceil((h - old_h) / 2))
#         if crop_bottom > 0:
#             img = img[..., :-crop_bottom, :]
#         return img



#     def pad_image(self, image):
#         h, w = image.shape[-2:]
#         new_h = int(np.ceil(h / self.patch_size) * self.patch_size)
#         new_w = int(np.ceil(w / self.patch_size) * self.patch_size)
#         img_padded = self.pad_with_new_size(image, (new_h, new_w), mode="replicate")
#         return img_padded, new_h, new_w


#     def postprocess_result(self, restored_image, window_sum, old_size):
#         restored_image = restored_image / (window_sum + 1e-8)
#         restored_image = self.crop_with_old_size(restored_image, old_size)
#         restored_image = torch.clamp(restored_image, 0.0, 1.0)
#         return restored_image


#     def get_patch_indices(self, new_h, new_w):
#         I_coords = np.arange(0, new_h - self.patch_size + 1, int(self.patch_size * (1 - self.overlap_percentage)))
#         J_coords = np.arange(0, new_w - self.patch_size + 1, int(self.patch_size * (1 - self.overlap_percentage)))
#         IJ_coords = np.meshgrid(I_coords, J_coords, indexing="ij")
#         IJ_coords = np.stack(IJ_coords).reshape(2, -1).T
#         n_blocks = len(I_coords) * len(J_coords)
#         return n_blocks, IJ_coords


#     def build_window(self, image_size, window_type):
#         H = W = image_size
#         if window_type == "kaiser":
#             window_i = torch.kaiser_window(H, beta=5, periodic=True)
#             window_j = torch.kaiser_window(W, beta=5, periodic=True)
#         elif window_type == "hann":
#             window_i = torch.hann_window(H, periodic=True)
#             window_j = torch.hann_window(W, periodic=True)
#         elif window_type == "hamming":
#             window_i = torch.hamming_window(H, periodic=True)
#             window_j = torch.hamming_window(W, periodic=True)
#         elif window_type == "bartlett":
#             window_i = torch.bartlett_window(H, periodic=True)
#             window_j = torch.bartlett_window(W, periodic=True)
#         else:
#             Exception("Window not implemented")

#         return window_i.unsqueeze(-1) * window_j.unsqueeze(0)



"""
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
@inproceedings{zhang2020deep,
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={0--0},
  year={2020}
}
# --------------------------------------------
"""


class ResUNet(nn.Module):
    def __init__(
        self,
        in_nc: int = 4,
        out_nc: int = 3,
        nc: tuple[int] = [64, 128, 256, 512],
        nb: int = 2,
        act_mode: str = "R",
        downsample_mode: Literal['avgpool', 'maxpool', 'strideconv'] = 'strideconv',
        upsample_mode: Literal['upconv', 'pixelshuffle', 'convtranspose'] = 'convtranspose'
    ):
        super().__init__()

        #  warning: different from ResUNet used by dpir
        self.m_head = conv(in_nc + 1, nc[0], bias=False, mode="C")

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError("downsample mode [{:s}] is not found".format(downsample_mode))

        self.m_down1 = sequential(
            *[
                ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[0], nc[1], bias=False, mode="2")
        )
        self.m_down2 = sequential(
            *[
                ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[1], nc[2], bias=False, mode="2")
        )
        self.m_down3 = sequential(
            *[
                ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[2], nc[3], bias=False, mode="2")
        )

        self.m_body  = sequential(
            *[
                ResBlock(nc[3], nc[3], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ]
        )

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv

        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle

        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose

        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )

        self.m_up3 = sequential(
            upsample_block(nc[3], nc[2], bias=False, mode="2"),
            *[
                ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ]
        )
        self.m_up2 = sequential(
            upsample_block(nc[2], nc[1], bias=False, mode="2"),
            *[
                ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ]
        )
        self.m_up1 = sequential(
            upsample_block(nc[1], nc[0], bias=False, mode="2"),
            *[
                ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ]
        )

        self.m_tail = conv(nc[0], out_nc, bias=False, mode="C")


    def forward(self, x: Tensor) -> Tensor:
        h, w = x.size()[-2:]
        modulo: int = 8
        mod_pad_h: int = (modulo - h % modulo) % modulo
        mod_pad_w: int = (modulo - w % modulo) % modulo
        x = nn.ReplicationPad2d((0, mod_pad_w, 0, mod_pad_h))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return x


