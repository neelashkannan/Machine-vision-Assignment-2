import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

def extract_sift_features(gray_images, subset_size=500):
    """
    Task 2.1: Extract SIFT features from grayscale images
    
    Args:
        gray_images: Array of grayscale images
        subset_size: Number of images to use for feature extraction
        
    Returns:
        all_descriptors: Array of SIFT descriptors
        keypoints_info: List containing the number of keypoints per image
        subset_indices: Indices of the subset images used
    """
    # Use a subset for feature extraction
    subset_indices = np.random.choice(len(gray_images), subset_size, replace=False)
    subset_images = gray_images[subset_indices]
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    all_descriptors = []
    keypoints_info = []
    
    for img in tqdm(subset_images, desc="Extracting SIFT features"):
        # Convert to uint8 if not already
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        # Some images might not have enough keypoints
        if descriptors is not None:
            all_descriptors.append(descriptors)
            keypoints_info.append(len(keypoints))
        else:
            keypoints_info.append(0)
    
    # Concatenate all descriptors
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
    else:
        all_descriptors = np.array([])
    
    return all_descriptors, keypoints_info, subset_indices

def visualize_keypoints(gray_images, labels, classes, output_path='output/sift_keypoints.png'):
    """
    Task 2.2: Visualize SIFT keypoints on sample images
    
    Args:
        gray_images: Array of grayscale images
        labels: Array of image labels
        classes: List of class names
        output_path: Path to save the visualization
        
    Returns:
        fig: Matplotlib figure with keypoints visualization
    """
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Pick a few images to visualize (one from each class if possible)
    sample_indices = []
    for i in range(10):  # For each class
        indices = np.where(labels == i)[0]
        if len(indices) > 0:
            sample_indices.append(indices[0])
    
    # If we couldn't find a sample from each class, add more from available classes
    while len(sample_indices) < 5 and len(sample_indices) < len(gray_images):
        sample_indices.append(np.random.choice(len(gray_images)))
    
    sample_indices = sample_indices[:5]  # Limit to 5 samples
    
    fig, axs = plt.subplots(1, len(sample_indices), figsize=(15, 5))
    
    for i, idx in enumerate(sample_indices):
        img = gray_images[idx]
        label = labels[idx]
        
        # Convert to uint8 if not already
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        # Detect keypoints
        keypoints, _ = sift.detectAndCompute(img, None)
        
        # Convert grayscale to RGB for visualization
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Draw keypoints
        img_with_keypoints = cv2.drawKeypoints(img_rgb, keypoints, None, 
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        axs[i].imshow(img_with_keypoints)
        axs[i].set_title(f"{classes[label]}: {len(keypoints)} keypoints")
        axs[i].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(output_path)
    
    return fig, keypoints