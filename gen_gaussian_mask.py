import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
import torch


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()
    return np.array(frames)


def binarize_alpha(frames, threshold=0.1):
    # frames: (T, H, W)
    return (frames > threshold).astype(np.float32)


def shift_binary_right(binary_mask, shift=5):
    """
    Shift the binarized mask to the right by a specified number of pixels.
    Parameter:
        binary_mask: binarized mask (T, H, W), the value is 0 or 1
        shift: Shift right by pixels (positive integer)
    Return:
        shifted_mask: Binary mask after right shift
    """
    T, H, W = binary_mask.shape
    shifted_mask = np.zeros_like(binary_mask)  

    for t in range(T): 
        frame = binary_mask[t]
        for h in range(H):  
            row = frame[h]  

            obj_cols = np.where(row == 1)[0]
            if len(obj_cols) == 0:
                continue  

            col_min = obj_cols.min()
            col_max = obj_cols.max()
            obj_width = col_max - col_min + 1

            new_col_min = col_min + shift  
            new_col_max = new_col_min + obj_width - 1  

            if new_col_max >= W:
                new_col_max = W - 1
                new_col_min = max(new_col_max - obj_width + 1, 0)

            if new_col_min < 0:
                new_col_min = 0
                new_col_max = min(obj_width - 1, W - 1)

            shifted_mask[t, h, new_col_min:new_col_max+1] = 1

    return shifted_mask


def create_gaussian_mask_from_ellipse(shape, mean, axes, angle,
                                      coverage_x=1.0, coverage_y=1.0,
                                      min_sigma=1e-2, clip_exp=-50):
    H, W = shape
    y, x = np.mgrid[0:H, 0:W]
    cx, cy = mean

    x_shifted = x - cx
    y_shifted = y - cy
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    x_rot = x_shifted * cos_t + y_shifted * sin_t
    y_rot = -x_shifted * sin_t + y_shifted * cos_t

    sigma_x = max((axes[0] / 2.0) * coverage_x, min_sigma)
    sigma_y = max((axes[1] / 2.0) * coverage_y, min_sigma)

    exponent = -0.5 * ((x_rot / sigma_x) **2 + (y_rot / sigma_y)** 2)
    exponent = np.clip(exponent, clip_exp, 0)
    gauss = np.exp(exponent)

    gmin, gmax = gauss.min(), gauss.max()
    if (gmax - gmin) < 1e-9:
        return np.zeros_like(gauss, dtype=np.float32)
    gauss = (gauss - gmin) / (gmax - gmin)
    return gauss.astype(np.float32)


def spatial_temporal_compress(frames):
    T, H, W = frames.shape
    n = (T - 1) // 4
    compressed = []
    first = cv2.resize(frames[0], (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    compressed.append(first)
    for i in range(n):
        group = frames[1 + 4*i : 1 + 4*(i+1)]
        group_resized = np.stack([
            cv2.resize(f, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
            for f in group
        ], axis=0)
        compressed.append(np.mean(group_resized, axis=0))
    return np.array(compressed)


def fit_geometric_ellipse(binary_mask, orient_threshold=0.9, shape_threshold=0.6, enlarge_factor=1.0, min_axis=2.0):
    kernel = np.ones((5,5), np.uint8)
    mask_shape = (binary_mask > shape_threshold).astype(np.uint8)
    if mask_shape.sum() == 0:
        return None, None, None

    ys, xs = np.where(mask_shape > 0)
    if len(xs) < 5:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        cx = float((x0 + x1) / 2.0)
        cy = float((y0 + y1) / 2.0)
        major = max(x1 - x0, min_axis)
        minor = max(y1 - y0, min_axis)
        angle = 0.0
        return (cx, cy), (major * enlarge_factor, minor * enlarge_factor), angle

    points = np.stack([xs, ys], axis=1).astype(np.float64)
    mean = points.mean(axis=0)

    cov = np.cov(points - mean, rowvar=False)
    if np.any(~np.isfinite(cov)) or cov.shape != (2, 2):
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        cx = float((x0 + x1) / 2.0)
        cy = float((y0 + y1) / 2.0)
        major = max(x1 - x0, min_axis)
        minor = max(y1 - y0, min_axis)
        angle = 0.0
        return (cx, cy), (major * enlarge_factor, minor * enlarge_factor), angle

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    coords = (points - mean) @ eigvecs
    half_major = max((coords[:,0].max() - coords[:,0].min()) / 2.0, min_axis/2.0)
    half_minor = max((coords[:,1].max() - coords[:,1].min()) / 2.0, min_axis/2.0)

    major = 2.0 * half_major * enlarge_factor
    minor = 2.0 * half_minor * enlarge_factor

    mask_orient = (binary_mask > orient_threshold).astype(np.uint8)
    if mask_orient.sum() >= 5:
        ys_o, xs_o = np.where(mask_orient > 0)
        pts_o = np.stack([xs_o, ys_o], axis=1).astype(np.float64)
        mean_o = pts_o.mean(axis=0)
        cov_o = np.cov(pts_o - mean_o, rowvar=False)
        if cov_o.shape == (2,2) and np.all(np.isfinite(cov_o)):
            eigvals_o, eigvecs_o = np.linalg.eigh(cov_o)
            order_o = np.argsort(eigvals_o)[::-1]
            eigvecs_o = eigvecs_o[:, order_o]
            angle = float(np.degrees(np.arctan2(eigvecs_o[1,0], eigvecs_o[0,0])))
        else:
            angle = float(np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0])))
    else:
        angle = float(np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0])))

    return (float(mean[0]), float(mean[1])), (float(max(major, min_axis)), float(max(minor, min_axis))), angle


def generate_best_aligned_gaussian(binary_mask,
                                   orient_threshold=0.9,
                                   shape_threshold=0.6,
                                   coverage_x=1.0,
                                   coverage_y=1.0,
                                   coverage_target=0.98,
                                   max_enlarge_iters=8,
                                   enlarge_step=1.12,
                                   debug=False):
    result = fit_geometric_ellipse(binary_mask,
                                   orient_threshold=orient_threshold,
                                   shape_threshold=shape_threshold,
                                   enlarge_factor=1.0)
    if result[0] is None:
        return np.zeros_like(binary_mask, dtype=np.float32), None, None, None

    mean, (major, minor), angle = result
    mask_shape = (binary_mask > shape_threshold).astype(np.uint8)
    total = mask_shape.sum()
    if total == 0:
        return np.zeros_like(binary_mask, dtype=np.float32), mean, (major, minor), angle

    cur_major, cur_minor = float(major), float(minor)
    best_cover = 0.0
    best_axes = (cur_major, cur_minor)
    for it in range(max_enlarge_iters):
        axes_cv = (int(max(1, round(cur_major / 2.0))), int(max(1, round(cur_minor / 2.0))))
        ellipse_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        cx_i, cy_i = int(round(mean[0])), int(round(mean[1]))
        cx_i = np.clip(cx_i, 0, binary_mask.shape[1]-1)
        cy_i = np.clip(cy_i, 0, binary_mask.shape[0]-1)

        try:
            cv2.ellipse(ellipse_mask,
                        center=(cx_i, cy_i),
                        axes=axes_cv,
                        angle=float(angle),
                        startAngle=0,
                        endAngle=360,
                        color=1,
                        thickness=-1)
        except Exception:
            break

        cover = float(np.logical_and(ellipse_mask > 0, mask_shape > 0).sum()) / float(total)
        if cover > best_cover:
            best_cover = cover
            best_axes = (cur_major, cur_minor)

        if cover >= coverage_target:
            break

        cur_major *= enlarge_step
        cur_minor *= enlarge_step

    final_axes = (best_axes[0] * 1.0, best_axes[1] * 1.0)
    gauss = create_gaussian_mask_from_ellipse(binary_mask.shape, mean, final_axes, angle,
                                              coverage_x=coverage_x, coverage_y=coverage_y)
    return gauss, mean, final_axes, angle


if __name__ == "__main__":
    video_path = "path/to/your/alpha_video.mp4"
    alpha_frames = read_video_frames(video_path)

    binary_alpha = binarize_alpha(alpha_frames, threshold=0.1) 
    # The number of moving pixels can be adjusted as needed.
    shift = int(binary_alpha.shape[1]/2)  # +: right, -: left 
    binary_alpha_shifted = shift_binary_right(binary_alpha, shift=shift)

    compressed_masks = spatial_temporal_compress(binary_alpha_shifted)

    # Adjustable as needed
    orient_threshold = 0.9
    shape_threshold = 0.2
    coverage_x, coverage_y = 1.0, 1.0

    gauss_info = []
    for m in compressed_masks:
        gauss, mean, axes, angle = generate_best_aligned_gaussian(
            m,
            orient_threshold=orient_threshold,
            shape_threshold=shape_threshold,
            coverage_x=coverage_x,
            coverage_y=coverage_y,
            coverage_target=0.98,
            max_enlarge_iters=10,
            enlarge_step=1.12,
            debug=True
        )
        gauss_info.append(gauss)

    torch.save(torch.tensor(gauss_info), "path/to/your/gauss_mask_shifted_right")