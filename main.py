import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

# Import task modules
import task1
import task2
import task3
import task5 as task4
import task6 

# Set page configuration
st.set_page_config(page_title="CIFAR-10 Image Classification", page_icon="🖼️", layout="wide")

# Create directories for outputs
os.makedirs('output', exist_ok=True)
os.makedirs('models', exist_ok=True)

# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Initialize session state variables
if 'trainset' not in st.session_state:
    st.session_state.trainset = None
if 'testset' not in st.session_state:
    st.session_state.testset = None
if 'gray_train_images' not in st.session_state:
    st.session_state.gray_train_images = None
if 'train_labels' not in st.session_state:
    st.session_state.train_labels = None
if 'gray_test_images' not in st.session_state:
    st.session_state.gray_test_images = None
if 'test_labels' not in st.session_state:
    st.session_state.test_labels = None
if 'descriptors' not in st.session_state:
    st.session_state.descriptors = None
if 'subset_indices' not in st.session_state:
    st.session_state.subset_indices = None
if 'codebook' not in st.session_state:
    st.session_state.codebook = None
if 'train_histograms' not in st.session_state:
    st.session_state.train_histograms = None
if 'test_histograms' not in st.session_state:
    st.session_state.test_histograms = None
if 'subset_train_labels' not in st.session_state:
    st.session_state.subset_train_labels = None
if 'svm_model' not in st.session_state:
    st.session_state.svm_model = None
if 'svm_metrics' not in st.session_state:
    st.session_state.svm_metrics = None
if 'cnn_model' not in st.session_state:
    st.session_state.cnn_model = None
if 'cnn_metrics' not in st.session_state:
    st.session_state.cnn_metrics = None
if 'augmented_images' not in st.session_state:
    st.session_state.augmented_images = None
if 'augmented_labels' not in st.session_state:
    st.session_state.augmented_labels = None
if 'aug_svm_model' not in st.session_state:
    st.session_state.aug_svm_model = None
if 'aug_svm_metrics' not in st.session_state:
    st.session_state.aug_svm_metrics = None
if 'aug_cnn_model' not in st.session_state:
    st.session_state.aug_cnn_model = None
if 'aug_cnn_metrics' not in st.session_state:
    st.session_state.aug_cnn_metrics = None
# Add current task to session state
if 'current_task' not in st.session_state:
    st.session_state.current_task = "Select a task..."
# Add task status to session state
if 'task_running' not in st.session_state:
    st.session_state.task_running = False

def on_task_select():
    """Callback when task is selected"""
    st.session_state.current_task = st.session_state.task_selector

def run_task1_1():
    """Load CIFAR-10 Dataset"""
    st.write("### Task 1.1: Load CIFAR-10 Dataset")
    
    with st.spinner("Loading CIFAR-10 dataset..."):
        progress_bar = st.progress(0)
        trainset, testset = task1.load_cifar10()
        progress_bar.progress(100)
        
        st.session_state.trainset = trainset
        st.session_state.testset = testset
    
    st.success(f"Dataset loaded successfully! Training set: {len(trainset)} images, Test set: {len(testset)} images")

def run_task1_2():
    """Visualize CIFAR-10 Samples"""
    st.write("### Task 1.2: Visualize CIFAR-10 Samples")
    
    if st.session_state.trainset is None:
        st.error("Please run Task 1.1 first to load the dataset")
        return
    
    with st.spinner("Visualizing samples from each class..."):
        fig = task1.visualize_samples(st.session_state.trainset)
        st.pyplot(fig)
    
    st.success("Visualization completed and saved to 'output/cifar10_samples.png'")

def run_task1_3():
    """Convert to Grayscale"""
    st.write("### Task 1.3: Convert to Grayscale")
    
    if st.session_state.trainset is None or st.session_state.testset is None:
        st.error("Please run Task 1.1 first to load the dataset")
        return
    
    with st.spinner("Converting images to grayscale..."):
        gray_train_images, train_labels, gray_test_images, test_labels, fig = task1.convert_to_grayscale(
            st.session_state.trainset, st.session_state.testset)
        
        st.session_state.gray_train_images = gray_train_images
        st.session_state.train_labels = train_labels
        st.session_state.gray_test_images = gray_test_images
        st.session_state.test_labels = test_labels
        
        st.pyplot(fig)
    
    st.success(f"Conversion completed! {len(gray_train_images)} training images and {len(gray_test_images)} test images converted to grayscale.")

def run_task2_1():
    """Extract SIFT Features"""
    st.write("### Task 2.1: Extract SIFT Features")
    
    if st.session_state.gray_train_images is None:
        st.error("Please run Task 1.3 first to convert images to grayscale")
        return
    
    subset_size = st.slider("Number of images for feature extraction", 100, 5000, 500, 100)
    
    with st.spinner(f"Extracting SIFT features from {subset_size} images..."):
        descriptors, keypoints_info, subset_indices = task2.extract_sift_features(
            st.session_state.gray_train_images, 
            subset_size
        )
        
        st.session_state.descriptors = descriptors
        st.session_state.keypoints_info = keypoints_info
        st.session_state.subset_indices = subset_indices
    
    if descriptors is not None and len(descriptors) > 0:
        st.success(f"Feature extraction completed! Extracted {len(descriptors)} SIFT descriptors.")
        st.write(f"Average number of keypoints per image: {np.mean(keypoints_info):.2f}")
    else:
        st.error("No descriptors could be extracted from the images.")

def run_task2_2():
    """Visualize SIFT Keypoints"""
    st.write("### Task 2.2: Visualize SIFT Keypoints")
    
    if st.session_state.gray_train_images is None:
        st.error("Please run Task 1.3 first to convert images to grayscale")
        return
    
    with st.spinner("Visualizing SIFT keypoints..."):
        fig, _ = task2.visualize_keypoints(
            st.session_state.gray_train_images, 
            st.session_state.train_labels, 
            CLASSES
        )
        st.pyplot(fig)
    
    st.success("Visualization completed and saved to 'output/sift_keypoints.png'")

def run_task3_1():
    """Generate Codebook"""
    st.write("### Task 3.1: Generate Codebook")
    
    if st.session_state.descriptors is None:
        st.error("Please run Task 2.1 first to extract SIFT features")
        return
    
    n_clusters = st.selectbox("Select codebook size", [50, 100, 200], index=1)
    force_retrain = st.checkbox("Force regenerate codebook (ignore saved model)", value=False)
    
    with st.spinner(f"Generating codebook with {n_clusters} visual words..."):
        kmeans, fig = task3.generate_codebook(
            st.session_state.descriptors, 
            n_clusters, 
            output_path=f'output/codebook_k{n_clusters}.png',
            force_retrain=force_retrain
        )
        
        st.session_state.codebook = kmeans
        if fig:
            st.pyplot(fig)
    
    if kmeans is not None:
        st.success(f"Codebook with {n_clusters} visual words generated successfully!")
    else:
        st.error("Failed to generate codebook")

def run_task3_2():
    """Create BoVW Histograms"""
    st.write("### Task 3.2: Create BoVW Histograms")
    
    if st.session_state.gray_train_images is None or st.session_state.codebook is None:
        st.error("Please run Task 3.1 first to generate the codebook")
        return
    
    subset_size = st.slider("Number of training images to use", 100, 5000, 500, 100)
    
    with st.spinner(f"Creating Bag of Visual Words histograms for {subset_size} training images..."):
        # Use the subset of images used for feature extraction
        subset_train_images = st.session_state.gray_train_images[st.session_state.subset_indices]
        subset_train_labels = st.session_state.train_labels[st.session_state.subset_indices]
        
        train_histograms, subset_train_labels, fig = task3.create_bovw_histograms(
            subset_train_images, 
            subset_train_labels, 
            st.session_state.codebook, 
            subset_size=None,  # No further subsetting needed
            output_path='output/bovw_histograms.png'
        )
        
        with st.spinner("Creating histograms for test images..."):
            test_histograms, test_labels, _ = task3.create_bovw_histograms(
                st.session_state.gray_test_images, 
                st.session_state.test_labels, 
                st.session_state.codebook
            )
        
        st.session_state.train_histograms = train_histograms
        st.session_state.subset_train_labels = subset_train_labels
        st.session_state.test_histograms = test_histograms
        
        if fig:
            st.pyplot(fig)
    
    st.success(f"Created {len(train_histograms)} training histograms and {len(test_histograms)} test histograms!")

def run_task3_3():
    """Compare Codebook Sizes"""
    st.write("### Task 3.3: Compare Codebook Sizes")
    
    if st.session_state.gray_train_images is None:
        st.error("Please run Task 1.3 first to convert images to grayscale")
        return
    
    with st.spinner("Comparing performance for different codebook sizes..."):
        sizes = [50, 100, 200]
        metrics_data, fig, table = task4.compare_codebook_sizes(
            st.session_state.gray_train_images,
            st.session_state.train_labels,
            st.session_state.gray_test_images,
            st.session_state.test_labels,
            sizes=sizes
        )
        
        st.pyplot(fig)
        
        # Create a DataFrame for better display
        data = []
        for size, metrics in metrics_data.items():
            data.append({
                'Codebook Size': size,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        
        df = pd.DataFrame(data)
        st.table(df)
    
    st.success("Codebook size comparison completed!")

def run_task4_1():
    """Train SVM Classifier"""
    st.write("### Task 4.1: Train SVM Classifier")
    
    if st.session_state.train_histograms is None or st.session_state.test_histograms is None:
        st.error("Please run Task 3.2 first to create BoVW histograms")
        return
    
    C = st.slider("SVM regularization parameter (C)", 0.1, 20.0, 10.0, 0.1)
    kernel = st.selectbox("SVM kernel", ["rbf", "linear", "poly"], index=0)
    force_retrain = st.checkbox("Force retraining (ignore saved model)", value=False)
    
    with st.spinner("Training/Loading SVM classifier..."):
        svm, metrics, predictions, fig = task4.train_svm_classifier(
            st.session_state.train_histograms,
            st.session_state.subset_train_labels,
            st.session_state.test_histograms,
            st.session_state.test_labels,
            C=C,
            kernel=kernel,
            force_retrain=force_retrain
        )
        
        st.session_state.svm_model = svm
        st.session_state.svm_metrics = metrics
        
        st.subheader("SVM Classification Results")
        st.write(f"Accuracy: {metrics['accuracy']:.4f}")
        st.write(f"Precision: {metrics['precision']:.4f}")
        st.write(f"Recall: {metrics['recall']:.4f}")
        st.write(f"F1-Score: {metrics['f1_score']:.4f}")
        
        st.pyplot(fig)
    
    st.success("SVM classifier training/loading completed!")

def run_task4_2():
    """Train CNN Classifier"""
    st.write("### Task 4.2: Train CNN Classifier")
    
    num_epochs = st.slider("Number of epochs", 1, 20, 5)
    batch_size = st.selectbox("Batch size", [32, 64, 128, 256], index=1)
    learning_rate = st.slider("Learning rate", 0.001, 0.1, 0.01, 0.001, format="%.3f")
    force_retrain = st.checkbox("Force retraining (ignore saved model)", value=False)
    
    with st.spinner("Preparing CIFAR-10 dataset for CNN..."):
        trainloader, valloader, testloader = task4.prepare_cifar10_for_cnn(batch_size)
    
    with st.spinner(f"Training/Loading CNN for {num_epochs} epochs..."):
        model, metrics, predictions, fig_curves, fig_cm = task4.train_cnn(
            trainloader, 
            valloader, 
            testloader, 
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            force_retrain=force_retrain
        )
        
        st.session_state.cnn_model = model
        st.session_state.cnn_metrics = metrics
        
        st.subheader("CNN Classification Results")
        st.write(f"Accuracy: {metrics['accuracy']:.4f}")
        st.write(f"Precision: {metrics['precision']:.4f}")
        st.write(f"Recall: {metrics['recall']:.4f}")
        st.write(f"F1-Score: {metrics['f1_score']:.4f}")
        
        st.pyplot(fig_curves)
        st.pyplot(fig_cm)
    
    st.success("CNN classifier training/loading completed!")

def run_task5_1():
    """Compare SVM and CNN"""
    st.write("### Task 5.1: Compare SVM and CNN")
    
    if st.session_state.svm_metrics is None or st.session_state.cnn_metrics is None:
        st.error("Please run Tasks 4.1 and 4.2 first to train the SVM and CNN classifiers")
        return
    
    with st.spinner("Comparing SVM and CNN performance..."):
        fig, improvement = task4.compare_models(
            st.session_state.svm_metrics,
            st.session_state.cnn_metrics
        )
        
        st.pyplot(fig)
        
        st.subheader("CNN Improvement over SVM")
        for metric, value in improvement.items():
            st.write(f"{metric.capitalize()}: {value:.2f}%")
    
    st.success("Model comparison completed!")

def run_task6_1():
    """Apply Data Augmentation"""
    st.write("### Task 6.1: Apply Data Augmentation")
    
    if st.session_state.gray_train_images is None:
        st.error("Please run Task 1.3 first to convert images to grayscale")
        return
    
    subset_size = st.slider("Number of images to augment", 100, 5000, 500, 100)
    
    with st.spinner(f"Applying data augmentation to {subset_size} images..."):
        augmented_images, augmented_labels, fig = task6.apply_data_augmentation(
            st.session_state.gray_train_images,
            st.session_state.train_labels,
            subset_size
        )
        
        st.session_state.augmented_images = augmented_images
        st.session_state.augmented_labels = augmented_labels
        
        st.pyplot(fig)
    
    st.success(f"Data augmentation completed! Created {len(augmented_images)} images from {subset_size} original images.")

def run_task6_2():
    """Train SVM with Augmented Data"""
    st.write("### Task 6.2: Train SVM with Augmented Data")
    
    if st.session_state.augmented_images is None:
        st.error("Please run Task 6.1 first to apply data augmentation")
        return
    
    force_retrain = st.checkbox("Force retraining (ignore saved model)", value=False)
    
    with st.spinner("Training/Loading SVM with augmented data..."):
        aug_svm, aug_svm_metrics, svm_metrics, fig = task6.train_svm_with_augmented_data(
            st.session_state.gray_train_images,
            st.session_state.train_labels,
            st.session_state.gray_test_images,
            st.session_state.test_labels,
            st.session_state.augmented_images,
            st.session_state.augmented_labels,
            force_retrain=force_retrain
        )
        
        st.session_state.aug_svm_model = aug_svm
        st.session_state.aug_svm_metrics = aug_svm_metrics
        
        # Display results
        st.subheader("SVM Classification Results with Augmented Data")
        st.write(f"Accuracy: {aug_svm_metrics['accuracy']:.4f}")
        st.write(f"Precision: {aug_svm_metrics['precision']:.4f}")
        st.write(f"Recall: {aug_svm_metrics['recall']:.4f}")
        st.write(f"F1-Score: {aug_svm_metrics['f1_score']:.4f}")
        
        st.pyplot(fig)
        
        # Display improvement
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        improvement = {}
        for metric in metrics:
            improvement[metric] = (aug_svm_metrics[metric] - svm_metrics[metric]) / svm_metrics[metric] * 100
        
        st.subheader("Improvement with Data Augmentation")
        for metric, value in improvement.items():
            st.write(f"{metric.capitalize()}: {value:.2f}%")
    
    st.success("SVM classifier with augmented data training/loading completed!")

def run_task6_3():
    """Train CNN with Augmentation"""
    st.write("### Task 6.3: Train CNN with Augmentation")
    
    num_epochs = st.slider("Number of epochs", 1, 20, 5)
    batch_size = st.selectbox("Batch size", [32, 64, 128, 256], index=1)
    learning_rate = st.slider("Learning rate", 0.001, 0.1, 0.01, 0.001, format="%.3f")
    force_retrain = st.checkbox("Force retraining (ignore saved model)", value=False)
    
    with st.spinner(f"Training/Loading CNN with augmentation for {num_epochs} epochs..."):
        model, metrics_aug, metrics_no_aug, fig_curves, fig_cm = task6.train_cnn_with_augmentation(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            force_retrain=force_retrain
        )
        
        st.session_state.aug_cnn_model = model
        st.session_state.aug_cnn_metrics = metrics_aug
        
        # Display results
        st.subheader("CNN Classification Results with Augmentation")
        st.write(f"Accuracy: {metrics_aug['accuracy']:.4f}")
        st.write(f"Precision: {metrics_aug['precision']:.4f}")
        st.write(f"Recall: {metrics_aug['recall']:.4f}")
        st.write(f"F1-Score: {metrics_aug['f1_score']:.4f}")
        
        st.pyplot(fig_curves)
        st.pyplot(fig_cm)
        
        # Display improvement
        metrics_list = ['accuracy', 'precision', 'recall', 'f1_score']
        improvement = {}
        for metric in metrics_list:
            improvement[metric] = (metrics_aug[metric] - metrics_no_aug[metric]) / metrics_no_aug[metric] * 100
        
        st.subheader("Improvement with Data Augmentation")
        for metric, value in improvement.items():
            st.write(f"{metric.capitalize()}: {value:.2f}%")
    
    st.success("CNN classifier with augmentation training/loading completed!")

def run_task7():
    """Final Comparison and Analysis"""
    st.write("### Task 7: Final Comparison and Analysis")
    
    if (st.session_state.svm_metrics is None or st.session_state.cnn_metrics is None or 
        st.session_state.aug_svm_metrics is None or st.session_state.aug_cnn_metrics is None):
        st.error("Please complete all model training tasks first")
        return
    
    with st.spinner("Creating final comparison..."):
        fig = task6.final_comparison(
            st.session_state.svm_metrics,
            st.session_state.aug_svm_metrics,
            st.session_state.cnn_metrics,
            st.session_state.aug_cnn_metrics
        )
        
        st.pyplot(fig)
        
        # Create comparison table
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'BoVW + SVM': [st.session_state.svm_metrics[m] for m in metrics],
            'BoVW + SVM with Augmentation': [st.session_state.aug_svm_metrics[m] for m in metrics],
            'CNN (ResNet-18)': [st.session_state.cnn_metrics[m] for m in metrics],
            'CNN with Augmentation': [st.session_state.aug_cnn_metrics[m] for m in metrics]
        }
        
        df = pd.DataFrame(data)
        st.table(df)
        
        # Find best model
        model_metrics = {
            'BoVW + SVM': st.session_state.svm_metrics,
            'BoVW + SVM with Augmentation': st.session_state.aug_svm_metrics,
            'CNN (ResNet-18)': st.session_state.cnn_metrics,
            'CNN with Augmentation': st.session_state.aug_cnn_metrics
        }
        
        best_model = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
        best_model_name = best_model[0]
        best_model_metrics = best_model[1]
        
        st.subheader("Summary of Findings")
        st.write(f"Best performing model: **{best_model_name}**")
        st.write(f"Best model accuracy: {best_model_metrics['accuracy']:.4f}")
        
        # Effect of data augmentation
        svm_improvement = ((st.session_state.aug_svm_metrics['accuracy'] - st.session_state.svm_metrics['accuracy']) / 
                          st.session_state.svm_metrics['accuracy'] * 100)
        
        cnn_improvement = ((st.session_state.aug_cnn_metrics['accuracy'] - st.session_state.cnn_metrics['accuracy']) / 
                          st.session_state.cnn_metrics['accuracy'] * 100)
        
        st.write("#### Effect of Data Augmentation:")
        st.write(f"SVM accuracy improvement: {svm_improvement:.2f}%")
        st.write(f"CNN accuracy improvement: {cnn_improvement:.2f}%")
        
        # Write conclusion
        st.subheader("Conclusion")
        st.write("""
        This project demonstrated the implementation of Bag of Visual Words (BoVW) and Convolutional Neural Network (CNN) 
        approaches for image classification on the CIFAR-10 dataset. The key findings include:
        
        1. CNN models significantly outperform traditional BoVW+SVM approaches
        2. Data augmentation improves performance for both approaches
        3. The combination of CNN with data augmentation achieves the best results
        
        This shows the power of deep learning approaches and the importance of data augmentation in improving model performance.
        """)
    
    st.success("Final analysis completed!")

def main():
    st.title("CIFAR-10 Image Classification with BoVW and CNN")
    st.markdown("""
    This application implements image classification on the CIFAR-10 dataset using:
    - Bag of Visual Words (BoVW) model with SVM classifier
    - Deep Learning approach with ResNet-18 CNN
    - Data augmentation techniques
    
    Follow the tasks step by step using the dropdown menu.
    """)
    
    # Create a dropdown for task selection that maintains its state
    task_options = [
        "Select a task...",
        "Task 1.1: Load CIFAR-10 Dataset",
        "Task 1.2: Visualize CIFAR-10 Samples",
        "Task 1.3: Convert to Grayscale",
        "Task 2.1: Extract SIFT Features",
        "Task 2.2: Visualize SIFT Keypoints",
        "Task 3.1: Generate Codebook",
        "Task 3.2: Create BoVW Histograms",
        "Task 3.3: Compare Codebook Sizes",  # New option
        "Task 4.1: Train SVM Classifier",
        "Task 4.2: Train CNN Classifier",
        "Task 5.1: Compare SVM and CNN",
        "Task 6.1: Apply Data Augmentation",
        "Task 6.2: Train SVM with Augmented Data",
        "Task 6.3: Train CNN with Augmentation",
        "Task 7: Final Comparison and Analysis"
    ]
    
    # Use callback to update current_task when dropdown changes
    st.selectbox(
        "Select a task to run:", 
        task_options, 
        key="task_selector",
        on_change=on_task_select,
        index=task_options.index(st.session_state.current_task)
    )
    
    # Check if CUDA is available
    import torch
    cuda_available = torch.cuda.is_available()
    st.sidebar.write(f"Using device: {'GPU (CUDA)' if cuda_available else 'cuda'}")
    
    # Run the selected task
    if st.button("Run Task") or st.session_state.task_running:
        st.session_state.task_running = True
        start_time = time.time()
        
        if st.session_state.current_task == "Task 1.1: Load CIFAR-10 Dataset":
            run_task1_1()
        elif st.session_state.current_task == "Task 1.2: Visualize CIFAR-10 Samples":
            run_task1_2()
        elif st.session_state.current_task == "Task 1.3: Convert to Grayscale":
            run_task1_3()
        elif st.session_state.current_task == "Task 2.1: Extract SIFT Features":
            run_task2_1()
        elif st.session_state.current_task == "Task 2.2: Visualize SIFT Keypoints":
            run_task2_2()
        elif st.session_state.current_task == "Task 3.1: Generate Codebook":
            run_task3_1()
        elif st.session_state.current_task == "Task 3.2: Create BoVW Histograms":
            run_task3_2()
        elif st.session_state.current_task == "Task 3.3: Compare Codebook Sizes":
            run_task3_3()
        elif st.session_state.current_task == "Task 4.1: Train SVM Classifier":
            run_task4_1()
        elif st.session_state.current_task == "Task 4.2: Train CNN Classifier":
            run_task4_2()
        elif st.session_state.current_task == "Task 5.1: Compare SVM and CNN":
            run_task5_1()
        elif st.session_state.current_task == "Task 6.1: Apply Data Augmentation":
            run_task6_1()
        elif st.session_state.current_task == "Task 6.2: Train SVM with Augmented Data":
            run_task6_2()
        elif st.session_state.current_task == "Task 6.3: Train CNN with Augmentation":
            run_task6_3()
        elif st.session_state.current_task == "Task 7: Final Comparison and Analysis":
            run_task7()
        else:
            st.info("Please select a task from the dropdown menu")
        
        end_time = time.time()
        st.sidebar.write(f"Task execution time: {end_time - start_time:.2f} seconds")
    
    # Show completed tasks in sidebar
    st.sidebar.write("### Completed Tasks:")
    
    completed_tasks = []
    
    if st.session_state.trainset is not None:
        completed_tasks.append("Task 1.1: Load CIFAR-10 Dataset ✓")
    
    if st.session_state.gray_train_images is not None:
        completed_tasks.append("Task 1.2: Visualize CIFAR-10 Samples ✓")
        completed_tasks.append("Task 1.3: Convert to Grayscale ✓")
    
    if st.session_state.descriptors is not None:
        completed_tasks.append("Task 2.1: Extract SIFT Features ✓")
    
    if st.session_state.codebook is not None:
        completed_tasks.append("Task 2.2: Visualize SIFT Keypoints ✓")
        completed_tasks.append("Task 3.1: Generate Codebook ✓")
    
    if st.session_state.train_histograms is not None:
        completed_tasks.append("Task 3.2: Create BoVW Histograms ✓")
    
    if st.session_state.svm_metrics is not None:
        completed_tasks.append("Task 4.1: Train SVM Classifier ✓")
    
    if st.session_state.cnn_metrics is not None:
        completed_tasks.append("Task 4.2: Train CNN Classifier ✓")
        completed_tasks.append("Task 5.1: Compare SVM and CNN ✓")
    
    if st.session_state.augmented_images is not None:
        completed_tasks.append("Task 6.1: Apply Data Augmentation ✓")
    
    if st.session_state.aug_svm_metrics is not None:
        completed_tasks.append("Task 6.2: Train SVM with Augmented Data ✓")
    
    if st.session_state.aug_cnn_metrics is not None:
        completed_tasks.append("Task 6.3: Train CNN with Augmentation ✓")
        completed_tasks.append("Task 7: Final Comparison and Analysis ✓")
    
    for task in completed_tasks:
        st.sidebar.write(task)
    
    # Add a button to reset task running state
    if st.session_state.task_running:
        if st.sidebar.button("Reset (Clear Task)"):
            st.session_state.task_running = False
            st.experimental_rerun()

if __name__ == "__main__":
    main()