# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import os
import os.path as osp
import av

import imageio
import torch
import torchvision

import numpy as np

__all__ = ['cache_video', 'cache_image', 'str2bool']

def create_alpha_video_pyav(frames, output_path, fps=30):
    """Creates a video with an alpha channel using PyAV."""
    height, width, _ = frames[0].shape
    container = av.open(output_path, mode='w')
    stream = container.add_stream('prores_ks', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuva444p10le'

    for frame in frames:
        av_frame = av.VideoFrame.from_ndarray(frame, format='rgba')
        for packet in stream.encode(av_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def cache_video(tensor_fgr,
                tensor_pha,
                save_file=None,
                fps=30,
                suffix='.mov',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
        # try:
        if True:
            # preprocess
            tensor_fgr = tensor_fgr.clamp(min(value_range), max(value_range))
            tensor_fgr = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor_fgr.unbind(2)
            ],
                                 dim=1).permute(1, 2, 3, 0)
            tensor_fgr = (tensor_fgr * 255).type(torch.uint8).cpu()

            tensor_pha = tensor_pha.clamp(min(value_range), max(value_range))
            tensor_pha = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor_pha.unbind(2)
            ],
                                 dim=1).permute(1, 2, 3, 0)
            tensor_pha = (tensor_pha * 255).type(torch.uint8).cpu()

            frames = []
            for frame_fgr, frame_pha in zip(tensor_fgr.numpy(), tensor_pha.numpy()):
                frame_pha = (0.0 + frame_pha[:,:,0:1] + frame_pha[:,:,1:2] + frame_pha[:,:,2:3]) / 3.
                frame = np.concatenate([frame_fgr, frame_pha.astype(np.uint8)], axis=2)
                frames.append(frame)


            create_alpha_video_pyav(frames, cache_file, fps)

            return cache_file
    else:
        print(f'cache_video failed, error: {error}', flush=True)
        return None


def cache_image(tensor,
                save_file,
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            torchvision.utils.save_image(
                tensor,
                save_file,
                nrow=nrow,
                normalize=normalize,
                value_range=value_range)
            return save_file
        except Exception as e:
            error = e
            continue


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')
