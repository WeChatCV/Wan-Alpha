import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
import torch


def spatial_temporal_compress(frames):
    """frames：(batch_size, T, H, W)"""
    batch_size, T, H, W = frames.shape
    compressed_batch = []  
    
    if torch.is_tensor(frames):
        frames = frames.detach()  
        if frames.dtype == torch.bfloat16:
            frames = frames.type(torch.float32)  
        frames_np = frames.cpu().numpy()  
    else:
        frames_np = frames 


    for b in range(batch_size):
        sample_frames = frames_np[b]  
        n = (T - 1) // 4  
        compressed = []
        
        first_frame = cv2.resize(
            sample_frames[0], 
            (W // 8, H // 8),  
            interpolation=cv2.INTER_AREA
        )
        compressed.append(first_frame)
        
        for i in range(n):
            group = sample_frames[1 + 4*i : 1 + 4*(i+1)]
            group_resized = np.stack([
                cv2.resize(f, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
                for f in group
            ], axis=0)

            compressed.append(np.mean(group_resized, axis=0))
        
        compressed_batch.append(np.array(compressed))
    
    return np.array(compressed_batch)


def gen_gauss_mask_for_video(alpha_frames):
    original_frames = alpha_frames[:, 0, :, :, :]
    batch_size, T, H, W = original_frames.shape
    H_c, W_c = H // 8, W // 8  

    if torch.is_tensor(original_frames):
        original_frames_np = original_frames.detach().type(torch.float32).cpu().numpy()
    else:
        original_frames_np = original_frames

    compressed_masks = spatial_temporal_compress(original_frames_np)

    mask = np.full_like(compressed_masks, 0, dtype=np.int8)
    
    mask[np.isclose(compressed_masks, 1.0)] = 1
    mask[np.isclose(compressed_masks, -1.0)] = -1


    for b in range(batch_size):

        cond_t0_compressed = ~np.isclose(compressed_masks[b, 0], 1.0) & \
                             ~np.isclose(compressed_masks[b, 0], -1.0)
        mask[b, 0][cond_t0_compressed] = -1

        n_groups = (T - 1) // 4
        for i in range(n_groups):
            t_compressed = i + 1  


            cond_compressed = ~np.isclose(compressed_masks[b, t_compressed], 1.0) & \
                              ~np.isclose(compressed_masks[b, t_compressed], -1.0)
            

            original_group = original_frames_np[b, 1 + 4*i : 1 + 4*(i+1)]
            
            original_group_cond = ~np.isclose(original_group, 1.0) & ~np.isclose(original_group, -1.0)
            
            all_4_frames_cond = np.all(original_group_cond, axis=0) # 形状 (H, W)

            reshaped = all_4_frames_cond.reshape(H_c, 8, W_c, 8)
            swapped = reshaped.swapaxes(1, 2)
            blocks = swapped.reshape(H_c, W_c, -1)
            cond_original_blocks = np.all(blocks, axis=2) # 形状 (H_c, W_c)

            final_cond = cond_compressed & cond_original_blocks

            mask[b, t_compressed][final_cond] = -1
            
    # print("mask.shape:", mask.shape)
    return torch.tensor(mask)