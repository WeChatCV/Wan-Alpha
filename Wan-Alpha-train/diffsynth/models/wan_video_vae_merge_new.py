import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
import sys

import imageio
from PIL import Image
import numpy as np
import cv2
import torchvision
import os
import random
from einops import rearrange, repeat
import tqdm

from diffsynth.models.wan_video_vae_merge import VideoVAE_, CausalConv3d, RMS_norm, AttentionBlock, count_conv3d, ResidualBlock, check_is_instance

CACHE_T = 2

class BlockWCausal(torch.nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim=192, dropout=0.0):
        super().__init__()
        self.conv1 = CausalConv3d(in_dim, mid_dim, 1)
        self.middle = torch.nn.Sequential(ResidualBlock(mid_dim, mid_dim, dropout),
                                          AttentionBlock(mid_dim),
                                          ResidualBlock(mid_dim, mid_dim, dropout),
                                          AttentionBlock(mid_dim),
                                          ResidualBlock(mid_dim, mid_dim, dropout),
                                          )
        self.head = torch.nn.Sequential(RMS_norm(mid_dim, images=False), torch.nn.SiLU(),
                                  CausalConv3d(mid_dim, out_dim, 3, padding=1))
    
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        
        ## head
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x

class Mapping(torch.nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim=192, dropout=0.0):
        super().__init__()
        self.block = BlockWCausal(in_dim, out_dim, mid_dim, dropout)
    
    def forward(self, x):
        self.conv_num = count_conv3d(self.block)
        self.conv_idx = [0]
        self.feat_map = [None] * self.conv_num

        for i in range(x.shape[2]):
            self.conv_idx = [0]
            if i == 0:
                out = self.block(x[:, :, :1, :, :],
                                   feat_cache=self.feat_map,
                                   feat_idx=self.conv_idx)
            else:
                out_ = self.block(x[:, :, i:i+1, :, :],
                                    feat_cache=self.feat_map,
                                    feat_idx=self.conv_idx)
                out = torch.cat([out, out_], 2)
        return out

class MergedWanVideoVAE(torch.nn.Module):

    def __init__(self, new_vae_path):
        super().__init__()

        vae_path = "Wan2.1-T2V-14B/Wan2.1_VAE.pth"
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]
        self.useless_scale = [torch.tensor([0] * 16), torch.tensor([1] * 16)]

        self.z_dim = 16

        # init model
        self.upsampling_factor = 8

        self.vae_fgr = VideoVAE_(z_dim=self.z_dim).eval().requires_grad_(False)
        self.middle = Mapping(192*4, 16, mid_dim=192*2)

        #### Load checkpoint

        # basic
        self.vae_fgr.load_state_dict(torch.load(vae_path, map_location=torch.device("cpu")))

        # middle
        state_dict = torch.load(new_vae_path, map_location='cpu')

        del self.vae_fgr.encoder.head
        del self.vae_fgr.conv1

        middle_state_dict = {}
        for key in state_dict:
            if key[:7] == "middle.":
                middle_state_dict[key[7:]] = state_dict[key]
        self.middle.load_state_dict(middle_state_dict)

        #####
        self.torch_dtype = torch.bfloat16
        self.vae_fgr.to(dtype=self.torch_dtype)
        self.middle.to(dtype=self.torch_dtype)
        self.eval()

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask
    
    def tiled_decode(self, hidden_states, device, tile_size, tile_stride):
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if (h-stride_h >= 0 and h-stride_h+size_h >= H): continue
            for w in range(0, W, stride_w):
                if (w-stride_w >= 0 and w-stride_w+size_w >= W): continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = T * 4 - 3
        weight = torch.zeros((1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)
        values = torch.zeros((1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)

        for h, h_, w, w_ in tqdm(tasks, desc="VAE decoding"):
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.decode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) * self.upsampling_factor, (size_w - stride_w) * self.upsampling_factor)
            ).to(dtype=hidden_states.dtype, device=data_device)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        values = values.clamp_(-1, 1)
        return values

    def tiled_encode(self, video_fgr, video_pha, device, tile_size, tile_stride):
        _, _, T, H, W = video_fgr.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if (h-stride_h >= 0 and h-stride_h+size_h >= H): continue
            for w in range(0, W, stride_w):
                if (w-stride_w >= 0 and w-stride_w+size_w >= W): continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = (T + 3) // 4
        weight = torch.zeros((1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=self.torch_dtype, device=data_device)
        values = torch.zeros((1, 16, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=self.torch_dtype, device=data_device)

        for h, h_, w, w_ in tasks:
            hidden_states_batch_fgr = video_fgr[:, :, :, h:h_, w:w_].to(computation_device, dtype=self.torch_dtype)
            hidden_states_batch_pha = video_pha[:, :, :, h:h_, w:w_].to(computation_device, dtype=self.torch_dtype)

            hidden_states_batch_fgr = self.vae_fgr.encode(hidden_states_batch_fgr, self.useless_scale)
            hidden_states_batch_pha = self.vae_fgr.encode(hidden_states_batch_pha, self.useless_scale)

            hidden_states_batch = self.middle(torch.cat([hidden_states_batch_fgr, hidden_states_batch_pha], dim=1)).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) // self.upsampling_factor, (size_w - stride_w) // self.upsampling_factor)
            ).to(dtype=self.torch_dtype, device=data_device)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        
        scale = self.scale
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=values.dtype, device=values.device) for s in scale]
            values = (values - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=values.dtype, device=values.device)
            values = (values - scale[0]) * scale[1]
        return values

    def single_encode(self, video_fgr, video_pha, device):
        video_fgr, video_pha = video_fgr.to(device, dtype=self.torch_dtype), video_pha.to(device, dtype=self.torch_dtype)

        mu_fgr = self.vae_fgr.encode(video_fgr, self.useless_scale)
        mu_pha = self.vae_fgr.encode(video_pha, self.useless_scale)
        mu = self.middle(torch.cat([mu_fgr, mu_pha], dim=1))
        print("mu_fgr", mu_fgr.shape, "mu_pha", mu_pha.shape, "mu", mu.shape)
        scale = self.scale
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
    
        return mu
    
    def single_decode(self, hidden_state, device):
        hidden_state = hidden_state.to(device)
        video = self.vae_fgr.decode(hidden_state, self.scale)
        return video.clamp_(-1, 1)
    
    def encode(self, videos_fgr, videos_pha, device, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        videos_fgr = [video.to("cpu") for video in videos_fgr]
        videos_pha = [video.to("cpu") for video in videos_pha]

        hidden_states = []
        for video_fgr, video_pha in zip(videos_fgr, videos_pha):
            video_fgr = video_fgr.unsqueeze(0)
            video_pha = video_pha.unsqueeze(0)
            if tiled:
                tile_size = (tile_size[0] * 8, tile_size[1] * 8)
                tile_stride = (tile_stride[0] * 8, tile_stride[1] * 8)
                hidden_state = self.tiled_encode(video_fgr, video_pha, device, tile_size, tile_stride)
            else:
                hidden_state = self.single_encode(video_fgr, video_pha, device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states
    
    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        if tiled:
            video = self.tiled_decode(hidden_states, device, tile_size, tile_stride)
        else:
            video = self.single_decode(hidden_states, device)
        return video