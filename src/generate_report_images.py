
import os
import cv2
import numpy as np
import MotionDetector as md

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datasets = ["Office", "RedChair", "EnterExitCrossingPaths2cor"]
    
    # Structure to hold results for visualization
    # results[dataset][spatial_param] = list of dicts with keys: filter, deriv, temporal_sigma, spatial_sigma, mask
    results = {}

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        dataset_path = os.path.join(base_dir, dataset)
        
        # Load Frames
        frames_gray = md.load_frames_and_grayscale(dataset_path)
        frames_color = md.load_frames_in_color(dataset_path)
        
        results[dataset] = {}
        
        # --- Box Filter Analysis ---
        for k_size in [3, 5]:
            results[dataset][f'{k_size}x{k_size}'] = []
            print(f"  Box Filter {k_size}x{k_size}:")
            
            # Apply Spatial Filter
            frames_spatial = md.apply_2d_spatial_filtering(frames_gray, std_dev=0, kernel_size=k_size, filter_type='box')
            
            # 1. Simple Temporal Derivative
            deriv_simple = md.apply_simple_diff_operator(frames_spatial)
            thresh = md.get_adaptive_threshold(deriv_simple, k=3.0)
            mask = md.form_binary_mask(deriv_simple, thresh)
            results[dataset][f'{k_size}x{k_size}'].append({
                'filter': 'box',
                'deriv': 'simple',
                'mask': mask
            })
            
            # 2. Gaussian Temporal Derivatives
            for t_sigma in [0.25, 0.5, 1.0]:
                deriv_gauss = md.apply_gaussian_diff_operator(frames_spatial, std_dev=t_sigma)
                thresh = md.get_adaptive_threshold(deriv_gauss, k=3.0)
                mask = md.form_binary_mask(deriv_gauss, thresh)
                results[dataset][f'{k_size}x{k_size}'].append({
                    'filter': 'box',
                    'deriv': 'gaussian',
                    'temporal_sigma': t_sigma,
                    'mask': mask
                })

        # --- Gaussian Spatial Filter Analysis ---
        # We will use 5x5 kernel size for Gaussian Spatial as implied by the visualize function
        k_size_gauss = 5
        results[dataset]['5x5'] = [] # Re-using key or creating new? The visualizer uses dataset['5x5']
        # Actually visualizer code: measurements = results[dataset]['5x5']
        # So I need to append to this list or make sure it handles both if I want (but visualizer logic is specific)
        # Visualizer separates by filter type loop.
        
        for s_sigma in [0.25, 0.5, 1.0]:
            print(f"  Gaussian Spatial Filter sigma={s_sigma}:")
            frames_spatial = md.apply_2d_spatial_filtering(frames_gray, std_dev=s_sigma, kernel_size=k_size_gauss, filter_type='gaussian')
            
            # 1. Simple Temporal Derivative
            deriv_simple = md.apply_simple_diff_operator(frames_spatial)
            thresh = md.get_adaptive_threshold(deriv_simple, k=3.0)
            mask = md.form_binary_mask(deriv_simple, thresh)
            results[dataset]['5x5'].append({
                'filter': 'gaussian',
                'deriv': 'simple',
                'spatial_sigma': s_sigma,
                'mask': mask
            })
            
            # 2. Gaussian Temporal Derivatives
            for t_sigma in [0.25, 0.5, 1.0]:
                deriv_gauss = md.apply_gaussian_diff_operator(frames_spatial, std_dev=t_sigma)
                thresh = md.get_adaptive_threshold(deriv_gauss, k=3.0)
                mask = md.form_binary_mask(deriv_gauss, thresh)
                results[dataset]['5x5'].append({
                    'filter': 'gaussian',
                    'deriv': 'gaussian',
                    'spatial_sigma': s_sigma,
                    'temporal_sigma': t_sigma,
                    'mask': mask
                })

        # Generate Visualizations
        # Try to find a good frame index. 
        # Office: ~55
        # RedChair: ~ ? (Need to check)
        # Crossing: ~ ?
        
        frame_idx = 55
        if dataset == 'RedChair':
             frame_idx = 55 # Guess
        elif dataset == 'EnterExitCrossingPaths2cor':
             frame_idx = 55 # Guess
             
        # Ensure frame_idx is within bounds
        if frame_idx >= len(frames_color):
            frame_idx = len(frames_color) // 2

        try:
            md.visualize_dataset_split_grids(dataset, output_dir, frames_color, results, frame_idx=frame_idx)
            if dataset == "Office":
                visualize_k_threshold_comparison(dataset, output_dir, frames_gray, frames_color, frame_idx=frame_idx)
            if dataset == "RedChair":
                visualize_redchair_ghosting(dataset, output_dir, frames_gray, frames_color, frame_idx=frame_idx)
        except Exception as e:
            print(f"Error visualizing {dataset}: {e}")

def visualize_k_threshold_comparison(dataset_name, output_dir, frames_gray, frames_color, frame_idx=55):
    """
    Generates comparison images for k=1 vs k=3 threshold values.
    Uses Box Filter lines to demonstrate noise sensitivity.
    """
    print(f"[i] Generating K-Value Comparison for: {dataset_name}...")
    
    # Use 3x3 Box Filter + Simple Derivative as baseline
    frames_spatial = md.apply_2d_spatial_filtering(frames_gray, std_dev=0, kernel_size=3, filter_type='box')
    deriv_simple = md.apply_simple_diff_operator(frames_spatial)
    
    # Compare k=1 vs k=3
    for k_val in [1.0, 3.0]:
        thresh = md.get_adaptive_threshold(deriv_simple, k=k_val)
        mask = md.form_binary_mask(deriv_simple, thresh)
        
        # Create Overlay
        orig = frames_color[frame_idx].copy()
        mask_frame = mask[frame_idx]
        overlay = orig.copy()
        overlay[mask_frame > 0] = [0, 0, 255] # Red overlay for high contrast
        combined = cv2.addWeighted(orig, 1.0, overlay, 0.5, 0)
        
        filename = f"{dataset_name}_box3x3_simple_k{int(k_val)}.png"
        cv2.imwrite(os.path.join(output_dir, filename), combined)
        print(f"    Saved {filename}")

def visualize_redchair_ghosting(dataset_name, output_dir, frames_gray, frames_color, frame_idx=100):
    """
    Generates comparison images for RedChair ghosting analysis.
    Varies Gaussian Spatial Sigma to show increased ghosting due to low frame rate.
    Uses Simple Temporal Derivative to isolate the spatial effect.
    """
    if dataset_name != "RedChair":
        return

    print(f"[i] Generating Ghosting Analysis for: {dataset_name}...")
    
    # Compare Spatial Sigma 0.25 vs 1.0 vs 3.0 (exaggerated)
    sigmas = [0.25, 1.0, 3.0]
    
    for s_sigma in sigmas:
        # High spatial smoothing
        frames_spatial = md.apply_2d_spatial_filtering(frames_gray, std_dev=s_sigma, kernel_size=5, filter_type='gaussian')
        
        # Simple Temporal Derivative 
        deriv_simple = md.apply_simple_diff_operator(frames_spatial)
        thresh = md.get_adaptive_threshold(deriv_simple, k=3.0)
        mask = md.form_binary_mask(deriv_simple, thresh)
        
        # Create Overlay
        orig = frames_color[frame_idx].copy()
        mask_frame = mask[frame_idx]
        overlay = orig.copy()
        overlay[mask_frame > 0] = [0, 0, 255] # Red overlay
        combined = cv2.addWeighted(orig, 1.0, overlay, 0.5, 0)
        
        filename = f"{dataset_name}_ghosting_sigma{s_sigma}.png"
        cv2.imwrite(os.path.join(output_dir, filename), combined)
        print(f"    Saved {filename}")

if __name__ == "__main__":
    main()
