# CIFAR-10 Image Classification: BoVW vs CNN Approaches with Data Augmentation

## Overview
This project implements and compares two distinct approaches to image classification on the CIFAR-10 dataset:
1. **Traditional Computer Vision**: Bag of Visual Words (BoVW) with SVM classifier
2. **Deep Learning**: Convolutional Neural Network (ResNet-18)

Both approaches are evaluated with and without data augmentation to analyze performance improvements.

## Key Features
- Complete implementation of the BoVW pipeline including SIFT feature extraction and codebook generation
- Fine-tuning of pre-trained ResNet-18 for CIFAR-10 classification
- Implementation of various data augmentation techniques
- Interactive Streamlit web application to visualize the entire classification pipeline
- Comprehensive performance comparison using accuracy, precision, recall, and F1-score

## Results Summary
- CNN (ResNet-18) significantly outperforms BoVW+SVM approach (82.73% vs 18.98% accuracy)
- Data augmentation improves BoVW+SVM performance by 2.32% (from 18.98% to 19.42%)
- Interestingly, data augmentation decreased CNN performance by 3.07% (from 82.73% to 80.19%)
- Medium-sized codebook (100 words) performed best for BoVW approach

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone [https://github.com/neelashkannan/Machine-vision-Assignement-2.git]
cd Machine-vision-Assignement-2
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit Application
Execute the following command to start the interactive web application:

```bash
streamlit run main.py
```

The application will open in your default web browser with a user interface that allows you to:
- Load and explore the CIFAR-10 dataset
- Convert images to grayscale
- Extract SIFT features
- Generate visual codebooks
- Create BoVW histograms
- Train SVM and CNN classifiers
- Apply data augmentation
- Compare model performances

### Step-by-step Execution
1. Select a task from the dropdown menu
2. Click the "Run Task" button
3. Follow the sequential workflow from Task 1.1 to Task 7
4. View completed tasks in the sidebar

## Project Structure
- `main.py`: Streamlit application with the complete workflow
- `task1.py`: Data loading, visualization, and grayscale conversion
- `task2.py`: SIFT feature extraction and keypoint visualization
- `task3.py`: Codebook generation and histogram creation for BoVW
- `task5.py`: SVM and CNN model training and evaluation functions
- `task6.py`: Data augmentation implementation and final comparison
- `requirements.txt`: Required Python packages
- `output/`: Directory for saved visualizations and results
- `models/`: Directory for saved trained models

## Implementation Details

### 1. Data Exploration and Preprocessing
- Loaded CIFAR-10 dataset (60,000 32×32 color images across 10 classes)
- Visualized samples from each class
- Converted RGB images to grayscale for feature extraction

### 2. Feature Extraction
- Implemented SIFT (Scale-Invariant Feature Transform) for keypoint detection
- Extracted 128-dimensional SIFT descriptors around keypoints
- Detected ~72 keypoints per image on average

### 3. Codebook Generation
- Applied k-means clustering to SIFT descriptors to create visual codebooks
- Experimented with different codebook sizes (50, 100, and 200 visual words)
- Created normalized histograms of visual words for each image

### 4. Classification Models
- **SVM Classifier**:
  - Trained SVM with RBF kernel on BoVW histograms
  - Optimized parameters (C=10.0, gamma='scale')
  - Evaluated performance on test set

- **CNN Classifier**:
  - Fine-tuned pre-trained ResNet-18 on CIFAR-10
  - Used standard data preprocessing and augmentation
  - Trained with cross-entropy loss and SGD optimizer

### 5. Data Augmentation
- Implemented various augmentation techniques:
  - Horizontal flipping
  - Rotation (±10 degrees)
  - Translation
  - Scaling
- Trained both models with augmented data and compared performance

## Author
Neelash Kannan A  
Heriot-Watt University  
Edinburgh, United Kingdom  
Email: na3018@hw.ac.uk

## Acknowledgments
- The CIFAR-10 dataset by Alex Krizhevsky and Geoffrey Hinton
- ResNet architecture by Kaiming He et al.
- Streamlit for the interactive web application framework
