import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import pickle

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)
os.makedirs('models', exist_ok=True)

def generate_codebook(descriptors, n_clusters=100, output_path=None, force_retrain=False):
    """
    Task 3.1: Generate codebook using k-means clustering
    
    Args:
        descriptors: Array of SIFT descriptors
        n_clusters: Number of clusters (visual words)
        output_path: Path to save the visualization
        force_retrain: Whether to force retraining even if a saved model exists
        
    Returns:
        kmeans: Trained KMeans model
        fig: Matplotlib figure with codebook visualization
    """
    if descriptors is None or len(descriptors) == 0:
        return None, None
    
    # Define model filename
    model_dir = 'models'
    model_filename = f"{model_dir}/kmeans_codebook_k{n_clusters}.pkl"
    
    # Check if model exists and load it if not forcing retrain
    if os.path.exists(model_filename) and not force_retrain:
        print(f"Loading existing codebook from {model_filename}")
        with open(model_filename, 'rb') as f:
            kmeans = pickle.load(f)
    else:
        print(f"Generating new codebook with {n_clusters} visual words...")
        # Use a subset of descriptors if there are too many
        max_descriptors = 100000  # Limit to save memory
        if len(descriptors) > max_descriptors:
            indices = np.random.choice(len(descriptors), max_descriptors, replace=False)
            sample_descriptors = descriptors[indices]
        else:
            sample_descriptors = descriptors
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(sample_descriptors)
        
        # Save the model
        with open(model_filename, 'wb') as f:
            pickle.dump(kmeans, f)
        print(f"Codebook saved to {model_filename}")
    
    # Visualize codebook (simplified 2D visualization) if output_path is provided
    fig = None
    if output_path:
        # Use a sample of descriptors for visualization
        max_viz_samples = 10000
        if len(descriptors) > max_viz_samples:
            viz_indices = np.random.choice(len(descriptors), max_viz_samples, replace=False)
            viz_descriptors = descriptors[viz_indices]
        else:
            viz_descriptors = descriptors
        
        # Use PCA to reduce dimensions to 2 for visualization
        pca = PCA(n_components=2)
        descriptors_2d = pca.fit_transform(viz_descriptors)
        centroids_2d = pca.transform(kmeans.cluster_centers_)
        
        # Get cluster assignments for the visualization sample
        cluster_labels = kmeans.predict(viz_descriptors)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(descriptors_2d[:, 0], descriptors_2d[:, 1], 
                            c=cluster_labels, cmap='viridis', 
                            alpha=0.5, s=10)
        
        # Plot centroids
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                  c='red', marker='X', s=100, label='Centroids')
        
        plt.title(f'Codebook Visualization (K={n_clusters})')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        
        # Save the figure
        fig.savefig(output_path)
    
    return kmeans, fig

def create_bovw_histograms(gray_images, labels, kmeans, subset_size=None, output_path=None):
    """
    Task 3.2: Create Bag of Visual Words histograms
    
    Args:
        gray_images: Array of grayscale images
        labels: Array of image labels
        kmeans: Trained KMeans model
        subset_size: Number of training images to use (None for all)
        output_path: Path to save the visualization
        
    Returns:
        histograms: BoVW histograms for images
        labels: Labels for histograms
        fig: Matplotlib figure with histograms visualization
    """
    n_clusters = kmeans.cluster_centers_.shape[0]
    
    # Use subset for faster processing
    if subset_size and subset_size < len(gray_images):
        subset_indices = np.random.choice(len(gray_images), subset_size, replace=False)
        subset_images = gray_images[subset_indices]
        subset_labels = labels[subset_indices]
    else:
        subset_images = gray_images
        subset_labels = labels
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Generate histograms
    histograms = []
    
    for img in tqdm(subset_images, desc="Creating BoVW histograms"):
        # Convert to uint8 if not already
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        # Create histogram of visual words
        histogram = np.zeros(n_clusters)
        
        if descriptors is not None:
            # Assign each descriptor to a cluster
            visual_words = kmeans.predict(descriptors)
            
            # Count frequency of each visual word
            for vw in visual_words:
                histogram[vw] += 1
                
            # Normalize histogram
            if np.sum(histogram) > 0:
                histogram /= np.sum(histogram)
        
        histograms.append(histogram)
    
    histograms = np.array(histograms)
    
    # Visualize some histograms
    fig = None
    if output_path:
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        
        for i in range(10):
            class_indices = np.where(subset_labels == i)[0]
            if len(class_indices) > 0:
                idx = class_indices[0]  # Get first image of each class
                row, col = i // 5, i % 5
                axs[row, col].bar(range(n_clusters), histograms[idx])
                axs[row, col].set_title(f"Class {i}")
                axs[row, col].set_xlabel('Visual Word ID')
                axs[row, col].set_ylabel('Frequency')
        
        plt.tight_layout()
        fig.savefig(output_path)
    
    return histograms, subset_labels, fig