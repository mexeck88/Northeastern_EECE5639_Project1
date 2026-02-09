""" MotionDetector.py

This file implements the requirements for Project 1 to be imported and called as a module within main.ipynb
There are four components to this operation:
    1. Load Frames
    2. convert frames to grayscale.
    3. Apply a 1-D differential operator at each pixel to compute a temporal derivative
    4. Threshold the abs values of the derivatives to create a 0 and 1 mask of the moving objects
    5. Combine the mask with the original frame to display the results.
    6. Save the frames back as a video file. (optional)
"""


# Imports
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import tqdm



def load_frames_and_grayscale(dir_path):
    """ 
    Load frames from a directory and stored the frames as a list of grayscale images.
    
    Args:
        dir_path (str): Path to the directory containing the frames.
    
    Returns:
        list: List of grayscale frames.
     """
    frames = []
    print("[i] Loading frames from directory: ", dir_path)

    files = sorted(os.listdir(dir_path))

    for filename in tqdm.tqdm(files, desc="Frames to Grayscale"):
        if filename.endswith(".jpg"):
            frame = cv2.imread(os.path.join(dir_path, filename))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frames.append(gray)                                
    return np.array(frames) # Shape: (Time, Height, Width)


def load_frames_in_color(dir_path):
    """ 
    Load frames from a directory and stored the frames as a list of color images.
    
    Args:
        dir_path (str): Path to the directory containing the frames.
    
    Returns:
        list: List of grayscale frames.
     """
    frames = []
    print("[i] Loading frames from directory: ", dir_path)

    files = sorted(os.listdir(dir_path))

    for filename in tqdm.tqdm(files, desc="Frames to color"):
        if filename.endswith(".jpg"):
            frame = cv2.imread(os.path.join(dir_path, filename))
            frames.append(frame)                                
    return np.array(frames) # Shape: (Time, Height, Width)


def apply_simple_diff_operator(frames):
    """
    Apply a simple difference operator to each frame to compute the temporal derivative for each pixel.
    
    Args:
        frames (numpy.ndarray): Array of grayscale frames with shape (Time, Height, Width).
    
    Returns:
        numpy.ndarray: Array of temporal derivatives with shape (Time, Height, Width).
    """

    # print(f"[i] Applying simple difference operator to frames [{frames.shape}]")
    simple_kernel = np.array([-0.5,0,0.5])
    deriv_simple = scipy.ndimage.convolve1d(frames, simple_kernel, axis=0)

    #trim the edges
    deriv_simple = deriv_simple[1:-1]

    return deriv_simple


def apply_gaussian_diff_operator(frames, std_dev):
    """
    Apply a Gaussian difference operator to each frame to compute the temporal derivative for each pixel.
    
    Args:
        frames (numpy.ndarray): Array of grayscale frames with shape (Time, Height, Width).
        std_dev (float): Standard deviation of the Gaussian kernel.
    
    Returns:
        numpy.ndarray: Array of temporal derivatives with shape (Time, Height, Width).
    """
    # print(f"[i] Applying Gaussian difference operator to frames [{frames.shape}], std dev: {std_dev}")
    gaussian_res = scipy.ndimage.gaussian_filter1d(frames, sigma=std_dev, axis=0, order=1)
    #trim the edges
    trim_ammount = int(4*std_dev)
    gaussian_res = gaussian_res[trim_ammount:-trim_ammount]
    return gaussian_res

def form_binary_mask(derivatives, threshold):
    """
    Form a binary mask of the moving objects by thresholding the absolute values of the derivatives.
    
    Args:
        derivatives (numpy.ndarray): Array of temporal derivatives with shape (Time, Height, Width).
        threshold (float): Threshold value for the binary mask.
    
    Returns:
        numpy.ndarray: Binary mask with shape (Time, Height, Width).
    """
    mask = np.abs(derivatives) > threshold
    return mask

def recombine_mask_with_color(frames, mask):
    """
    Recombine the mask with the original frames to display the results. The mask will be presented as a transparent red overlay on the original frames.
    
    Args:
        frames (numpy.ndarray): Array of grayscale frames with shape (Time, Height, Width).
        mask (numpy.ndarray): Binary mask with shape (Time, Height, Width).
    
    Returns:
        numpy.ndarray: Array of recombined frames with shape (Time, Height, Width).
    """
    t_frames = frames.shape[0]
    t_mask = mask.shape[0]
    
    if t_frames > t_mask:
        diff = t_frames - t_mask
        start = diff // 2
        end = start + t_mask
        frames_trimmed = frames[start:end]
    else:
        frames_trimmed = frames
    #print(f"[i] Recombining mask {mask.shape} with trimmed frames {frames_trimmed.shape}")
    if frames_trimmed.dtype != np.uint8:
        frames_uint8 = cv2.normalize(frames_trimmed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        frames_uint8 = frames_trimmed
        
    # Check if we are already in color (T, H, W, 3) or grayscale (T, H, W)
    if len(frames_uint8.shape) == 4 and frames_uint8.shape[3] == 3:
        frames_bgr = frames_uint8
    else:
        frames_bgr = np.stack((frames_uint8,) * 3, axis=-1)

    # Apply a bright green overlay for the mask
    alpha = 0.5
    overlay = np.zeros_like(frames_bgr)
    
    # Assign Bright GREEN color [0, 255, 0] for BGR to the masked areas
    overlay[mask > 0] = [0, 255, 0] 
    
    # Additive blending: Original (1.0) + Overlay (alpha)
    recombined_frames = cv2.addWeighted(frames_bgr, 1.0, overlay, alpha, 0)
    return recombined_frames

def apply_2d_spatial_filtering(frames, std_dev, kernel_size, filter_type='gaussian'):
    """
    Apply a 2D spatial smoothing filter to each frame.
    
    Args:
        frames (numpy.ndarray): Array of grayscale frames with shape (Time, Height, Width).
        std_dev (float): Standard deviation of the Gaussian kernel. (Ignored if filter_type='box')
        kernel_size (int): Size of the kernel (N x N).
        filter_type (str): 'gaussian' or 'box'. Default is 'gaussian'.
    
    Returns:
        numpy.ndarray: Array of filtered frames with shape (Time, Height, Width).
    """
    if kernel_size not in [3, 5]:
         # You might want to relax this check if you want to test larger kernels, 
         # but the assignment said 3x3 and 5x5 specifically.
         raise ValueError("kernel_size must be 3 or 5")
    
    print(f"[i] Applying 2D spatial filtering ({filter_type}) to frames [{frames.shape}], size: {kernel_size}, sigma: {std_dev}")
    # Prepare Kernel (or use cv2 directly)
    if filter_type == 'gaussian':
        # Create 2D Gaussian kernel manually as before
        kernel = cv2.getGaussianKernel(kernel_size, std_dev)
        kernel = kernel @ kernel.T
    elif filter_type == 'box':
        # Box filter kernel is just ones / area
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")
    # Apply 2D filter to each frame
    # cv2.filter2D works for any custom kernel
    filtered_img = cv2.filter2D(src=frames, ddepth=-1, kernel=kernel)
    
    return filtered_img

def get_adaptive_threshold(derivatives, k=3.0):
    """
    Calculate an adaptive threshold based on the standard deviation of the temporal derivatives.
    
    Args:
        derivatives (numpy.ndarray): Array of temporal derivatives with shape (Time, Height, Width).
        k (float): Factor to multiply the standard deviation by to determine the threshold.
    
    Returns:
        float: Adaptive threshold value.
    """
    # Calculate std dev of the temporal derivatives (assuming mean is ~0)
    noise_sigma = np.std(derivatives)
    
    # Threshold is k * sigma (e.g., 3 standard deviations)
    return k * noise_sigma

def save_as_video(frames, output_path, fps=30, codec='avc1'):
    """
    Save the array of frames as a video file.
    
    Args:
        frames (numpy.ndarray): Array of frames with shape (Time, Height, Width, 3) or (Time, Height, Width).
                                Must be uint8 (0-255).
        output_path (str): Path to save the video.
        fps (int): Frames per second.
    """
    if len(frames) == 0:
        print("[!] No frames to save.")
        return

    # Check if frames are grayscale or color
    if len(frames.shape) == 3: # (Time, Height, Width) -> Grayscale
        height, width = frames.shape[1], frames.shape[2]
        is_color = False
    else: # (Time, Height, Width, 3) -> Color
        height, width = frames.shape[1], frames.shape[2]
        is_color = True

    # Determine codec
    if codec:
        fourcc = cv2.VideoWriter_fourcc(*codec)
    elif output_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Default for mp4 if not specified
    else:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Default fallbak

    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=is_color)
    print(f"[i] Saving video to {output_path} ({len(frames)} frames, {width}x{height}, Color={is_color}, Codec={codec or 'default'})")

    for i in range(len(frames)):
        # Ensure frame is uint8
        frame = frames[i].astype(np.uint8)
        
        # If we said isColor=True but have grayscale, convert it
        if is_color and len(frame.shape) == 2:
             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
             
        out.write(frame)

    out.release()
    print("[i] Video saved successfully.")

def visualize_dataset_split_grids(dataset_name, output_dir, original_frames_color, results, frame_idx=55):
    """
    This function generates the final grid images to compare all of our different approach variations

    Args:
        dataset_name (str): Name of the dataset.
        original_frames_color (numpy.ndarray): Array of original color frames with shape (Time, Height, Width, 3).
        frame_idx (int): Index of the frame to visualize.

    Returns:
        None
    """
    print(f"[i] Processing & Saving Results for: {dataset_name}...")
    
    # Define Column Labels
    col_labels = ["Simple Diff", r"Gauss Diff ($\sigma_t=0.25$)", r"Gauss Diff ($\sigma_t=0.5$)", r"Gauss Diff ($\sigma_t=1.0$)"]
    # Helper: Create Overlay & Save Individual File
    def process_and_save(mask, filename_suffix):
        # Create Overlay
        orig = original_frames_color[frame_idx].copy()
        mask_frame = mask[frame_idx]
        overlay = orig.copy()
        overlay[mask_frame > 0] = [0, 255, 0] # Bright Green
        combined = cv2.addWeighted(orig, 1.0, overlay, 0.4, 0)
        
        # Save Individual File (BGR for cv2.imwrite)
        filename = f"{dataset_name}_{filename_suffix}.png"
        cv2.imwrite(os.path.join(output_dir, filename), combined)
        
        # Return RGB for Plotting
        return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    # --- FIGURE 1: BOX FILTER ANALYSIS (2x4) ---
    rows_box = [("Box Filter (3x3)", '3x3'), ("Box Filter (5x5)", '5x5')]
    
    fig1, axes1 = plt.subplots(len(rows_box), 4, figsize=(20, 8))
    fig1.suptitle(f"{dataset_name.capitalize()} - Box Spatial Filter Analysis", fontsize=18)
    
    for r, (row_label, k_size) in enumerate(rows_box):
        measurements = results[dataset_name][k_size]
        
        # Col 0: Simple Diff
        try:
            res = next(m for m in measurements if m['filter'] == 'box' and m['deriv'] == 'simple')
            img = process_and_save(res['mask'], f"box_{k_size}_simple_diff")
            axes1[r, 0].imshow(img)
        except StopIteration: pass
        axes1[r, 0].set_ylabel(row_label, fontsize=14, rotation=90, labelpad=10)
        
        # Cols 1-3: Gaussian Temporal diffs
        for c, t_sigma in enumerate([0.25, 0.5, 1.0]):
            try:
                res = next(m for m in measurements 
                           if m['filter'] == 'box' and m['deriv'] == 'gaussian' and m['temporal_sigma'] == t_sigma)
                img = process_and_save(res['mask'], f"box_{k_size}_gaussT{t_sigma}")
                axes1[r, c+1].imshow(img)
            except StopIteration: pass
    # Labels Fig 1
    for c, title in enumerate(col_labels): axes1[0, c].set_title(title, fontsize=14)
    for ax in axes1.flatten(): ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_box_grid.png", dpi=150)
    plt.show()
    # --- FIGURE 2: GAUSSIAN SPATIAL FILTER ANALYSIS (3x4) ---
    rows_gauss = [
        (r"Gaussian ($\sigma_s=0.25$)", 0.25),
        (r"Gaussian ($\sigma_s=0.5$)", 0.5),
        (r"Gaussian ($\sigma_s=1.0$)", 1.0),
    ]
    fig2, axes2 = plt.subplots(len(rows_gauss), 4, figsize=(20, 12))
    fig2.suptitle(f"{dataset_name.capitalize()} - Gaussian Spatial Filter Analysis", fontsize=18)
    
    for r, (row_label, s_sigma) in enumerate(rows_gauss):
        measurements = results[dataset_name]['5x5'] # Focusing on 5x5 kernel for Gaussian Spatial
        
        # Col 0: Simple Diff
        try:
            res = next(m for m in measurements 
                       if m['filter'] == 'gaussian' and m['deriv'] == 'simple' and m['spatial_sigma'] == s_sigma)
            img = process_and_save(res['mask'], f"gaussS{s_sigma}_simple_diff")
            axes2[r, 0].imshow(img)
        except StopIteration: pass
        axes2[r, 0].set_ylabel(row_label, fontsize=14, rotation=90, labelpad=10)
        
        # Cols 1-3: Gaussian Temporal Diffs
        for c, t_sigma in enumerate([0.25, 0.5, 1.0]):
            try:
                res = next(m for m in measurements 
                           if m['filter'] == 'gaussian' and m['deriv'] == 'gaussian' 
                           and m['spatial_sigma'] == s_sigma and m['temporal_sigma'] == t_sigma)
                img = process_and_save(res['mask'], f"gaussS{s_sigma}_gaussT{t_sigma}")
                axes2[r, c+1].imshow(img)
            except StopIteration: pass
    # Labels Fig 2
    for c, title in enumerate(col_labels): axes2[0, c].set_title(title, fontsize=14)
    for ax in axes2.flatten(): ax.set_xticks([]); ax.set_yticks([]) 
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_gaussian_grid.png", dpi=150)
    plt.show()
