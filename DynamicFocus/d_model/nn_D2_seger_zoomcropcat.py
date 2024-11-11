from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from d_model.nn_B0_deformed_sampler import get_grid_Bx2xHSxWS, deformed_unsampler, int_rount_scale_grid
from d_model.nn_B1_croper import get_idxs_crop4
from d_model.nn_D5_seger_crop import SegerCrop

from utility.torch_tools import gen_grid_mtx_2xHxW
from d_model.nn_B4_Unet6 import UNet6


class SegerZoomCropCat(nn.Module):

    def __init__(self, base_module_deformation: Type[nn.Module], base_module: Type[nn.Module], in_channels=4, out_channels=1, downsample_factor=8, kernel_gridsp=4 + 1, kernel_gblur=16 + 1, cbase_seg0=4, cbase_seg=32, nlayer_seg0=2,
                 nlayer_seg=4, priori=True):
        super(SegerZoomCropCat, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = out_channels

        self.downsample_factor = downsample_factor
        self.downsample_degree = int(np.log2(self.downsample_factor))

        self.alias = self.downsample_degree + 1

        self.kernel_gridsp = kernel_gridsp

        self.pad_gridsp = self.kernel_gridsp // 2

        self.gen_seg_0 = base_module_deformation(in_channels=(in_channels + 1) * self.alias, out_channels=out_channels, base_channels=cbase_seg0, layer_num=nlayer_seg0)
        self.gen_seg = base_module(in_channels=(in_channels + 1) * self.alias, out_channels=out_channels, base_channels=cbase_seg, layer_num=nlayer_seg)

        self.ave_pool = nn.AvgPool2d(kernel_size=self.downsample_factor, stride=self.downsample_factor, padding=0)

        sigma = kernel_gblur // 2
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_gblur, sigma=(sigma, sigma))

        self.max_pool = nn.MaxPool2d(kernel_size=self.downsample_factor, stride=self.downsample_factor, padding=0)

        self.ave_pool_d = nn.ModuleList()
        self.max_pool_d = nn.ModuleList()

        self.ave_pool_d.append(nn.Identity())
        self.max_pool_d.append(nn.Identity())

        for ds_degree in range(1, self.downsample_degree + 1):
            cur_downsample_factor = 2 ** (ds_degree)

            cur_ave_pool = nn.AvgPool2d(kernel_size=cur_downsample_factor, stride=cur_downsample_factor, padding=0)
            cur_max_pool = nn.MaxPool2d(kernel_size=cur_downsample_factor, stride=cur_downsample_factor, padding=0)

            self.ave_pool_d.append(cur_ave_pool)
            self.max_pool_d.append(cur_max_pool)

        self.priori = priori

        self.transform_rgbaf = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5], std=[0.229, 0.224, 0.225, 0.25, 0.25])  # 针对 ImageNet 进行标准化
        ])

    def get_idxs_crop_BxAx4(self, hidx_B: torch.Tensor, widx_B: torch.Tensor, H, W):
        assert hidx_B.shape == widx_B.shape
        B, = hidx_B.shape

        idxs_crop_BxAx4_s = []
        for b in range(B):
            idxs_crop_Ax4_s = []
            for a in range(self.alias):
                cur_ds_factor = 2 ** a
                pad = get_idxs_crop4(hidx_B[b].item(), widx_B[b].item(), H, W, H // cur_ds_factor, W // cur_ds_factor)
                idxs_crop_Ax4_s.append(pad)
            idxs_crop_BxAx4_s.append(idxs_crop_Ax4_s)

        idxs_crop_BxAx4 = torch.tensor(idxs_crop_BxAx4_s, dtype=torch.int64)
        return idxs_crop_BxAx4

    def gen_alias(self, any_BxCxHxW: torch.Tensor, idxs_crop_BxAx4: torch.Tensor, fctns: nn.ModuleList):
        B, C, H, W = any_BxCxHxW.shape
        target_device = any_BxCxHxW.device
        A = len(fctns)
        HS = H // self.downsample_factor
        WS = W // self.downsample_factor

        any_as_BxAxCxHSxWS = torch.zeros((B, A, C, HS, WS), dtype=torch.float32, device=target_device)

        for b in range(B):
            for a in range(self.alias):
                left, right, up, down = idxs_crop_BxAx4[b, a]

                any_as_BxAxCxHSxWS[b, a] = fctns[-(a + 1)](any_BxCxHxW[b, :, up:down, left:right])

        return any_as_BxAxCxHSxWS

    def gen_unalias(self, any_as_BxAxCxHSxWS: torch.Tensor, idxs_crop_BxAx4: torch.Tensor):

        # plt_multi_imgshow([x_ds_rgbaf_C1xHSxWS[:-1] for x_ds_rgbaf_C1xHSxWS in x_ds_rgbaf_BxAxC1xHSxWS[0]], row_col=(2, 2))
        # plt_show()

        B, A, C, HS, WS = any_as_BxAxCxHSxWS.shape
        target_device = any_as_BxAxCxHSxWS.device

        H = HS * self.downsample_factor
        W = WS * self.downsample_factor

        any_ua_BxCxHxW = torch.zeros((B, C, H, W), dtype=torch.float32, device=target_device)

        for a in range(A):
            cur_H = H // (2 ** a)
            cur_W = W // (2 ** a)

            any_as_curA_BxCxHcxWc = F.interpolate(any_as_BxAxCxHSxWS[:, a, :, :, :], (cur_H, cur_W), mode='bilinear', align_corners=True)

            for b in range(B):
                left, right, up, down = idxs_crop_BxAx4[b, a]

                any_ua_BxCxHxW[b, :, up:down, left:right] = any_as_curA_BxCxHcxWc[b]

            # plt_imgshow(any_ua_BxCxHxW[0, :-1, :, :])
            # plt_show()

        return any_ua_BxCxHxW

    def forward(self, x_BxCxHxW: torch.Tensor, x_Bx2: torch.Tensor):
        B, C, H, W = x_BxCxHxW.shape

        HS = H // self.downsample_factor  # 64
        WS = W // self.downsample_factor  # 128

        target_device = x_BxCxHxW.device

        # prepare focus map
        hidx_B = (x_Bx2[:, 0] * (H - 1))
        widx_B = (x_Bx2[:, 1] * (W - 1))
        grid_mtx_Bx2xHxW = gen_grid_mtx_2xHxW(H, W, device=target_device).unsqueeze(0).repeat(B, 1, 1, 1)
        dist_BxHxW = torch.sqrt((grid_mtx_Bx2xHxW[:, 0, :, :] - hidx_B[:, None, None]) ** 2 + (grid_mtx_Bx2xHxW[:, 1, :, :] - widx_B[:, None, None]) ** 2)
        dist_norm_Bx1xHxW = (dist_BxHxW / torch.amax(dist_BxHxW, dim=[-1, -2], keepdim=True)).unsqueeze(1)

        # make priori
        priori_Bx1xHSxWS = None
        init_value = 3.0
        area = 20
        if self.priori:
            priori_Bx1xHSxWS = torch.zeros(B, 1, HS, WS).to(dtype=torch.float32, device=target_device)

            for bidx, (hidx_1, widx_1) in enumerate(zip(hidx_B.to(int).tolist(), widx_B.to(int).tolist())):
                left, right, up, bottom = get_idxs_crop4(hidx_1, widx_1, HS, WS, area, area)
                priori_Bx1xHSxWS[bidx, :, up:bottom, left:right] = init_value
            priori_Bx1xHSxWS = priori_Bx1xHSxWS * 2 - init_value

        # assign focus map
        idxs_crop_BxAx4 = self.get_idxs_crop_BxAx4(hidx_B, widx_B, H, W).to(device=target_device)

        x_as_rgbaf_BxAxC1xHSxWS = torch.zeros(B, self.alias, C + 1, HS, WS).to(dtype=torch.float32, device=x_BxCxHxW.device)
        B, A, C1, HS, WS = x_as_rgbaf_BxAxC1xHSxWS.shape
        x_as_rgbaf_BxAxC1xHSxWS[:, :, :-1, :, :] = self.gen_alias(x_BxCxHxW, idxs_crop_BxAx4, self.ave_pool_d)
        x_as_rgbaf_BxAxC1xHSxWS[:, :, -1:, :, :] = self.gen_alias(dist_norm_Bx1xHxW, idxs_crop_BxAx4, self.max_pool_d)

        x_as_rgbaf_BxAC1xHSxWS = self.transform_rgbaf(x_as_rgbaf_BxAxC1xHSxWS.flatten(start_dim=0, end_dim=1)).view(B, A * (C + 1), HS, WS)

        out0_Bx1xHSxWS = self.gen_seg_0(x_as_rgbaf_BxAC1xHSxWS)
        if priori_Bx1xHSxWS is not None:
            out0_Bx1xHSxWS = out0_Bx1xHSxWS + priori_Bx1xHSxWS

        # gen density map

        dmap_pred_ds_Bx1xHSxWS = torch.sigmoid(out0_Bx1xHSxWS) * x_as_rgbaf_BxAxC1xHSxWS[:, 0, -2:-1, :, :]

        # gen grid
        dm_v_Bx1xHSPxWSP = F.pad(self.gaussian_blur(dmap_pred_ds_Bx1xHSxWS), (self.pad_gridsp, self.pad_gridsp, self.pad_gridsp, self.pad_gridsp), mode='replicate')
        grid_pred_BxHSxWSx2 = get_grid_Bx2xHSxWS(dm_v_Bx1xHSPxWSP, HS, WS, kernel_size=self.kernel_gridsp)

        x_gs_rgbaf_BxAxC1xHSxWS = torch.zeros(B, A, C + 1, HS, WS).to(dtype=torch.float32, device=x_BxCxHxW.device)

        for b in range(B):
            for a in range(self.alias):
                left, right, up, down = idxs_crop_BxAx4[b, a]

                x_gs_rgbaf_BxAxC1xHSxWS[b:b + 1, a, :-1] = F.grid_sample(x_BxCxHxW[b:b + 1, :, up:down, left:right], grid_pred_BxHSxWSx2[b:b + 1], mode='bilinear', align_corners=True)
                x_gs_rgbaf_BxAxC1xHSxWS[b:b + 1, a, -1:] = F.grid_sample(dist_norm_Bx1xHxW[b:b + 1, :, up:down, left:right], grid_pred_BxHSxWSx2[b:b + 1], mode='bilinear', align_corners=True)

        x_gs_rgbaf_BxAC1xHSxWS = self.transform_rgbaf(x_gs_rgbaf_BxAxC1xHSxWS.flatten(start_dim=0, end_dim=1)).view(B, A * (C + 1), HS, WS)

        out1_Bx1xHSxWS = self.gen_seg(x_gs_rgbaf_BxAC1xHSxWS)

        if priori_Bx1xHSxWS is not None:
            out1_Bx1xHSxWS = out1_Bx1xHSxWS  # + priori_Bx1xHSxWS

        y_pred_gs_Bx1xHSxWS = torch.sigmoid(out1_Bx1xHSxWS) * x_gs_rgbaf_BxAxC1xHSxWS[:, 0, -2:-1, :, :]

        return y_pred_gs_Bx1xHSxWS, grid_pred_BxHSxWSx2, dmap_pred_ds_Bx1xHSxWS, x_as_rgbaf_BxAxC1xHSxWS, x_gs_rgbaf_BxAxC1xHSxWS

    def downsample_x_real_BxCxHSxWS(self, x_BxCxHxW: torch.Tensor):
        B, C, H, W = x_BxCxHxW.shape
        HS = H // self.downsample_factor  # 64
        WS = W // self.downsample_factor  # 128
        return F.interpolate(x_BxCxHxW, (HS, WS), mode='bilinear', align_corners=True)

    def downsample_y_maxpool_real_Bx1xHSxWS(self, y_real_Bx1xHxW: torch.Tensor):
        return self.max_pool(y_real_Bx1xHxW)

    def downsample_y_gridsp_real_Bx1xHSxWS(self, y_real_Bx1xHxW: torch.Tensor, grid_pred_BxHSxWSx2: torch.Tensor):
        return F.grid_sample(y_real_Bx1xHxW, grid_pred_BxHSxWSx2, mode='bilinear', align_corners=True)

    def output_y_pred_Bx1xHxW(self, y_pred_Bx1xHSxWS: torch.Tensor, grid_pred_BxHSxWSx2: torch.Tensor):
        B, _, HS, WS = y_pred_Bx1xHSxWS.shape
        H = HS * self.downsample_factor
        W = WS * self.downsample_factor

        grid_pred_BxHSxWSx2 = torch.flip(grid_pred_BxHSxWSx2, dims=[-1])
        grid_pred_Bx2xHSxWS = grid_pred_BxHSxWSx2.permute(0, 3, 1, 2)
        grid_pred_Bx2xHSxWS = int_rount_scale_grid(grid_pred_Bx2xHSxWS, H, W)

        return deformed_unsampler(y_pred_Bx1xHSxWS, grid_pred_Bx2xHSxWS, H, W)


if __name__ == '__main__':
    pass
    in_channels = 4
    out_channels = 1
    canvas_H = 640
    canvas_W = 640
    sample_factor = 8
    B = 5

    x_BxCxHxW = torch.randn(B, in_channels, canvas_H, canvas_W)
    x_Bx2 = torch.randn(B, 2)
    e2e = SegerZoomCropCat(base_module=UNet6, in_channels=in_channels, out_channels=out_channels, downsample_factor=sample_factor)

    y_pred_gs_Bx1xHSxWS, grid_pred_BxHSxWSx2, dmap_pred_ds_Bx1xHSxWS, x_as_rgbaf_BxAxC1xHSxWS, x_gs_rgbaf_BxAxC1xHSxWS = e2e(x_BxCxHxW, x_Bx2)

    # plt_multi_imgshow(imgs=[label_x_BxKxHxW[0, :, :, 0], label_x_BxKxHxW[0, :, :, 1]], titles=['w', 'h'], row_col=(2, 1))
    # plt.show(block=True)
