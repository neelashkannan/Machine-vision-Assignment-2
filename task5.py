import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import os
import pickle

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)
os.makedirs('models', exist_ok=True)

# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda")

def train_svm_classifier(train_histograms, train_labels, test_histograms, test_labels, 
                         C=10.0, kernel='rbf', force_retrain=False, output_path='output/svm_confusion_matrix.png'):
    """
    Task 4.1: Train SVM classifier on BoVW histograms or load pre-trained model
    
    Args:
        train_histograms: BoVW histograms for training
        train_labels: Labels for training histograms
        test_histograms: BoVW histograms for testing
        test_labels: Labels for test histograms
        C: SVM regularization parameter
        kernel: SVM kernel type
        force_retrain: Whether to force retraining even if a saved model exists
        output_path: Path to save the confusion matrix
        
    Returns:
        svm: Trained SVM model
        metrics: Dictionary containing performance metrics
        predictions: Predicted labels for test data
        fig: Matplotlib figure with confusion matrix
    """
    print(f"Train histograms shape: {train_histograms.shape}")
    print(f"Test histograms shape: {test_histograms.shape}")
    
    # Define model filename based on parameters and feature count
    model_dir = 'models'
    n_features = train_histograms.shape[1]
    model_filename = f"{model_dir}/svm_model_C{C}_{kernel}_features{n_features}.pkl"
    
    # Check if model file exists and load it if not forcing retrain
    if os.path.exists(model_filename) and not force_retrain:
        print(f"Loading existing SVM model from {model_filename}")
        with open(model_filename, 'rb') as f:
            svm = pickle.load(f)
            
        # Verify the model's feature count matches our histograms
        if hasattr(svm, 'shape_fit_') and svm.shape_fit_[1] != n_features:
            print(f"Warning: Loaded model expects {svm.shape_fit_[1]} features but our histograms have {n_features} features.")
            print("Training a new model instead.")
            # Fall through to training a new model
        else:
            print("Successfully loaded compatible model.")
    else:
        print("Model not found or force retrain enabled, training new model.")
    
    # Train new model if needed
    if not os.path.exists(model_filename) or force_retrain or (hasattr(svm, 'shape_fit_') and svm.shape_fit_[1] != n_features):
        print(f"Training new SVM model with {n_features} features...")
        # Train SVM
        svm = SVC(kernel=kernel, C=C, gamma='scale', probability=True)
        svm.fit(train_histograms, train_labels)
        
        # Save the model
        with open(model_filename, 'wb') as f:
            pickle.dump(svm, f)
        print(f"SVM model saved to {model_filename}")
    
    # Predict on test set
    predictions = svm.predict(test_histograms)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Create a confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - SVM')
    
    # Save the figure
    fig.savefig(output_path)
    
    return svm, metrics, predictions, fig

def prepare_cifar10_for_cnn(batch_size=64):
    """
    Prepare CIFAR-10 dataset for CNN training
    
    Args:
        batch_size: Batch size for data loaders
        
    Returns:
        trainloader: DataLoader for training set
        valloader: DataLoader for validation set
        testloader: DataLoader for test set
    """
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Just normalization for validation/test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Split training data into training and validation sets
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, valloader, testloader

def train_cnn(trainloader, valloader, testloader, num_epochs=5, learning_rate=0.01, 
             force_retrain=False, output_path_curves='output/resnet_training_curves.png',
             output_path_cm='output/cnn_confusion_matrix.png'):
    """
    Task 4.2: Train CNN classifier or load pre-trained model
    
    Args:
        trainloader: DataLoader for training set
        valloader: DataLoader for validation set
        testloader: DataLoader for test set
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        force_retrain: Whether to force retraining even if a saved model exists
        output_path_curves: Path to save the training curves
        output_path_cm: Path to save the confusion matrix
        
    Returns:
        model: Trained CNN model
        metrics: Dictionary containing performance metrics
        predictions: Predicted labels for test data
        fig_curves: Figure with training curves
        fig_cm: Figure with confusion matrix
    """
    # Define model filename based on parameters
    model_dir = 'models'
    model_filename = f"{model_dir}/resnet18_e{num_epochs}_lr{learning_rate:.4f}.pth"
    
    # Load pre-trained ResNet-18 model
    model = resnet18(pretrained=True)
    
    # Modify the final fully connected layer for 10 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    # Check if saved model exists and load it if not forcing retrain
    if os.path.exists(model_filename) and not force_retrain:
        print(f"Loading existing CNN model from {model_filename}")
        model.load_state_dict(torch.load(model_filename, map_location=device))
        # Load training history if available
        history_filename = f"{model_dir}/resnet18_history_e{num_epochs}_lr{learning_rate:.4f}.pkl"
        if os.path.exists(history_filename):
            with open(history_filename, 'rb') as f:
                history = pickle.load(f)
                train_losses = history['train_losses']
                val_losses = history['val_losses']
                train_accs = history['train_accs']
                val_accs = history['val_accs']
        else:
            # Create dummy history if not available
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
    else:
        print(f"Training new CNN model for {num_epochs} epochs...")
        # Move model to device
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Training loop
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = running_loss / len(trainloader.dataset)
            train_acc = 100. * correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation phase
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(valloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Track loss and accuracy
                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_loss = running_loss / len(valloader.dataset)
            val_acc = 100. * correct / total
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                print(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}% - saving model...")
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_filename)
            
            # Update learning rate
            scheduler.step()
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        with open(f"{model_dir}/resnet18_history_e{num_epochs}_lr{learning_rate:.4f}.pkl", 'wb') as f:
            pickle.dump(history, f)
    
    # Plot training curves
    fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    if train_losses:  # Only plot if we have training history
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(train_accs, label='Train Accuracy')
        ax2.plot(val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
    else:
        ax1.text(0.5, 0.5, 'No training history available', ha='center', va='center')
        ax2.text(0.5, 0.5, 'No training history available', ha='center', va='center')
    
    plt.tight_layout()
    fig_curves.savefig(output_path_curves)
    
    # Ensure model is on the correct device and in eval mode for testing
    model = model.to(device)
    model.eval()
    
    # Evaluate on test set
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Testing CNN"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Create a confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    fig_cm, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - CNN')
    
    # Save the figure
    fig_cm.savefig(output_path_cm)
    
    return model, metrics, predictions, fig_curves, fig_cm

def compare_models(svm_metrics, cnn_metrics, output_path='output/model_comparison.png'):
    """
    Task 5.1: Compare SVM and CNN model performance
    
    Args:
        svm_metrics: Dictionary containing SVM performance metrics
        cnn_metrics: Dictionary containing CNN performance metrics
        output_path: Path to save the comparison plot
        
    Returns:
        fig: Matplotlib figure with model comparison
        improvement: Dictionary containing improvement percentages
    """
    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, [svm_metrics[m] for m in metrics], width, label='BoVW + SVM')
    ax.bar(x + width/2, [cnn_metrics[m] for m in metrics], width, label='CNN (ResNet-18)')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_path)
    
    # Calculate improvement percentages
    improvement = {}
    for metric in metrics:
        improvement[metric] = (cnn_metrics[metric] - svm_metrics[metric]) / svm_metrics[metric] * 100
    
    return fig, improvement

def compare_codebook_sizes(gray_train_images, train_labels, gray_test_images, test_labels, 
                          sizes=[50, 100, 200], C=10.0, kernel='rbf', output_path='output/codebook_comparison.png'):
    """
    Compare the impact of different codebook sizes on SVM performance
    
    Args:
        gray_train_images: Grayscale training images
        train_labels: Training labels
        gray_test_images: Grayscale test images
        test_labels: Test labels
        sizes: List of codebook sizes to compare
        C: SVM regularization parameter
        kernel: SVM kernel type
        output_path: Path to save the comparison plot
        
    Returns:
        metrics_data: Dictionary containing performance metrics for each size
        fig: Matplotlib figure with comparison
        table: Formatted table as a string
    """
    import task2
    import task3
    
    metrics_data = {}
    
    for size in sizes:
        print(f"\n--- Processing codebook size {size} ---")
        
        # Step 1: Extract SIFT features
        print("Extracting SIFT features...")
        subset_size = 500  # You can adjust this
        descriptors, _, subset_indices = task2.extract_sift_features(
            gray_train_images, subset_size=subset_size)
        
        # Step 2: Generate codebook
        print(f"Generating codebook with {size} visual words...")
        kmeans, _ = task3.generate_codebook(
            descriptors, n_clusters=size, force_retrain=True)
        
        # Step 3: Create BoVW histograms
        print("Creating BoVW histograms...")
        # Use the subset of images used for feature extraction
        subset_train_images = gray_train_images[subset_indices]
        subset_train_labels = train_labels[subset_indices]
        
        train_histograms, subset_train_labels, _ = task3.create_bovw_histograms(
            subset_train_images, subset_train_labels, kmeans)
        
        test_histograms, test_labels_subset, _ = task3.create_bovw_histograms(
            gray_test_images, test_labels, kmeans)
        
        # Step 4: Train SVM and get metrics
        print("Training SVM classifier...")
        _, metrics, _, _ = train_svm_classifier(
            train_histograms, subset_train_labels, 
            test_histograms, test_labels_subset,
            C=C, kernel=kernel, force_retrain=True)
        
        metrics_data[size] = metrics
        print(f"Size {size} metrics: {metrics}")
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(metrics_list))
    width = 0.25
    
    for i, (size, metrics) in enumerate(metrics_data.items()):
        offset = (i - len(sizes)/2 + 0.5) * width
        ax.bar(x + offset, [metrics[m] for m in metrics_list], width, label=f'Codebook Size {size}')
    
    ax.set_ylabel('Score')
    ax.set_title('Impact of Codebook Size on BoVW+SVM Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_path)
    
    # Create a formatted table as a string
    table = "TABLE II: Impact of codebook size on BoVW+SVM performance\n"
    table += "=" * 60 + "\n"
    table += f"{'Codebook Size':^15}{'Accuracy':^15}{'Precision':^15}{'Recall':^15}{'F1-Score':^15}\n"
    table += "-" * 60 + "\n"
    
    for size in sizes:
        metrics = metrics_data[size]
        table += f"{size:^15}{metrics['accuracy']:^15.4f}{metrics['precision']:^15.4f}"
        table += f"{metrics['recall']:^15.4f}{metrics['f1_score']:^15.4f}\n"
    
    table += "=" * 60
    
    print("\n" + table)
    
    # Also save the table to a text file
    with open('output/codebook_comparison_table.txt', 'w') as f:
        f.write(table)
    
    return metrics_data, fig, table