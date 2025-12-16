# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_image, cache_video, str2bool

import numpy as np
from PIL import Image, ImageDraw
import imageio
import cv2
import torchvision
import zipfile


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."

    # The default sampling steps are 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50

    if args.sample_shift is None:
        args.sample_shift = 5.0

    # The default number of frames are 81 for text-to-video tasks.
    if args.frame_num is None:
        args.frame_num = 81

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an RGBA video from a text prompt using Wan-Alpha"
    )

    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        help="The area (width*height) of the generated video."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--vae_lora_checkpoint",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.")
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.")
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None."
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--lora_ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None)
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--n_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        help="The negative prompt to generate the image or video from. ")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--lora_prefix",
        type=str,
        default="",)
    parser.add_argument(
        "--lightx2v_path",
        type=str,
        default=None,
        help="The path to the lightx2v checkpoint.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def render_video(tensor_fgr,
                tensor_pha,
                nrow=8,
                normalize=True,
                value_range=(-1, 1)):
    
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
    frames_fgr = []
    frames_pha = []
    for frame_fgr, frame_pha in zip(tensor_fgr.numpy(), tensor_pha.numpy()):
        frame_pha = (0.0 + frame_pha[:,:,0:1] + frame_pha[:,:,1:2] + frame_pha[:,:,2:3]) / 3.
        frame = np.concatenate([frame_fgr[:,:,::-1], frame_pha.astype(np.uint8)], axis=2)
        frames.append(frame)
        frames_fgr.append(frame_fgr)
        frames_pha.append(frame_pha)

    def create_checkerboard(size=30, pattern_size=(830, 480), color1=(140, 140, 140), color2=(113, 113, 113)):
        img = Image.new('RGB', (pattern_size[0], pattern_size[1]), color1)
        draw = ImageDraw.Draw(img)
        for i in range(0, pattern_size[0], size):
            for j in range(0, pattern_size[1], size):
                if (i + j) // size % 2 == 0:
                    draw.rectangle([i, j, i+size, j+size], fill=color2)
        return img

    def blender_background(frame_rgba, checkerboard):
        alpha_channel = frame_rgba[:, :, 3:] / 255. 
        checkerboard = np.array(checkerboard)
        checkerboard = cv2.resize(checkerboard, (frame_rgba.shape[1], frame_rgba.shape[0]))

        frame_rgb = frame_rgba[:, :, :3] * alpha_channel + checkerboard * (1-alpha_channel)
        return frame_rgb.astype(np.uint8)[:,:,::-1]
    
    checkerboard = create_checkerboard()
    video_checkerboard = [blender_background(f, checkerboard) for f in frames]

    return video_checkerboard, frames

def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = WAN_CONFIGS['t2v-14B']
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]


    logging.info("Creating Wan-Alpha pipeline.")
    wan_t2v = wan.WanT2V_dora_lightx2v(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        vae_lora_checkpoint=args.vae_lora_checkpoint,
        lora_path=args.lora_path,
        lightx2v_path=args.lightx2v_path,
        lora_ratio=args.lora_ratio,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    def save_video(save_path, target_frames, fps=16):
        writer = imageio.get_writer(
            save_path, fps=fps, codec='libx264', quality=8)
        for frame in target_frames:
            writer.append_data(frame)
        writer.close()

    logging.info(
        "Generating video...")

    with open(args.prompt_file) as fin:
        all_prompts = fin.read().splitlines()
    
    os.makedirs(args.output_dir, exist_ok=True)

    for args.prompt in all_prompts:
        args.prompt = args.lora_prefix + args.prompt
        logging.info(f"Input prompt: {args.prompt}")

        videos = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            n_prompt=args.n_prompt,
            offload_model=args.offload_model)

        if rank == 0:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                    "_")[:50]
            suffix = '.mov'
            args.save_file = f"{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix
            args.save_file = os.path.join(args.output_dir, args.save_file)

            logging.info(f"Saving generated video to {args.save_file}")

            video_fgr, video_pha = videos
            video_checkerboard, frames = render_video(video_fgr[None], video_pha[None])

            os.makedirs(f"{args.output_dir}/{formatted_time}", exist_ok=True)
            save_video(f"{args.output_dir}/{formatted_time}/checkerboard.mp4", video_checkerboard)
            zip_path = f"{args.output_dir}/{formatted_time}/pngs.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for idx, img in enumerate(frames):
                        success, buffer = cv2.imencode(".png", img)
                        if not success:
                            print(f"Failed to encode image {idx}, skipping...")
                            continue
                        
                        filename = f"img_{idx:03d}.png"
                        zipf.writestr(filename, buffer.tobytes())

    exit()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
