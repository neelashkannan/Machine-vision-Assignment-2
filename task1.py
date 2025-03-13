import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import os

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_cifar10():
    """
    Task 1.1: Load CIFAR-10 dataset
    
    Returns:
        trainset: Training dataset
        testset: Test dataset
    """
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    return trainset, testset

def visualize_samples(trainset, output_path='output/cifar10_samples.png'):
    """
    Task 1.2: Visualize samples from each class in CIFAR-10
    
    Args:
        trainset: Training dataset
        output_path: Path to save the visualization
        
    Returns:
        fig: Matplotlib figure with the visualization
    """
    fig, axs = plt.subplots(10, 5, figsize=(15, 25))
    
    for i in range(10):
        # Find indices for this class
        indices = [j for j, (img, label) in enumerate(trainset) if label == i]
        selected_idx = np.random.choice(indices, 5)
        
        for j, idx in enumerate(selected_idx):
            img, label = trainset[idx]
            img = img.numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            
            axs[i, j].imshow(img)
            axs[i, j].set_title(CLASSES[label])
            axs[i, j].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(output_path)
    
    return fig

def convert_to_grayscale(trainset, testset, output_path='output/cifar10_grayscale_samples.png'):
    """
    Task 1.3: Convert images to grayscale
    
    Args:
        trainset: Training dataset
        testset: Test dataset
        output_path: Path to save the visualization
        
    Returns:
        gray_train_images: Grayscale training images
        train_labels: Training labels
        gray_test_images: Grayscale test images
        test_labels: Test labels
        fig: Matplotlib figure with grayscale visualization
    """
    # Process training set
    gray_train_images = []
    train_labels = []
    
    for img, label in trainset:
        img = img.numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_train_images.append(gray_img)
        train_labels.append(label)
    
    # Process test set
    gray_test_images = []
    test_labels = []
    
    for img, label in testset:
        img = img.numpy().transpose((1, 2, 0))
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_test_images.append(gray_img)
        test_labels.append(label)
    
    # Visualize some grayscale images
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        idx = np.where(np.array(train_labels) == i)[0][0]  # Get first image of each class
        row, col = i // 5, i % 5
        axs[row, col].imshow(gray_train_images[idx], cmap='gray')
        axs[row, col].set_title(CLASSES[train_labels[idx]])
        axs[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(output_path)
    
    return np.array(gray_train_images), np.array(train_labels), np.array(gray_test_images), np.array(test_labels), fig