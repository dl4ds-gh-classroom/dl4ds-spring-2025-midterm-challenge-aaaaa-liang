import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import wandb
import json

# Import eval_cifar100 and eval_ood to ensure tests run correctly on SCC
from eval_cifar100 import evaluate_cifar100_test
from eval_ood import evaluate_ood_test, create_ood_df

################################################################################
# Model Definition - Simple CNN 
################################################################################
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # Add an extra conv layer after low accuracy 
        
        # Pooling 
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers 
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 100) # Define 100 classes for CIFAR-100 
        
        # ReLU activations 
        self.relu = nn.ReLU()
        # Drop out for regularization 
        self.dropout = nn.Dropout(0.5)
        
        # Batch normalization layers 
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
    
    def forward(self, x):
        # Forward pass
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))  # Forward pass through extra conv layer 
        
        # Flatten 
        x = x.view(-1, 512 * 2 * 2)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

################################################################################
# Data Transformation - Modified 
################################################################################
transform_train = transforms.Compose([
    # Random changes for variation 
    transforms.RandomHorizontalFlip(), # Flip images to horizontal  
    transforms.RandomRotation(15), # Rotate images 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Change colors 
    transforms.RandomCrop(32, padding=4), # Crop images with padding 
    transforms.ToTensor(), # Convert images to tensors 
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), # Normalize based on CIFAR-100 statistics 
])

transform_test = transforms.Compose([
    transforms.ToTensor(), # Convert images to tensors 
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # Normalize based on CIFAR-100 statistics 
])

################################################################################
# Configuration Dictionary - Modified 
################################################################################
def main():
    CONFIG = {
        "model": "SimpleCNN",
        "batch_size": 32, # Increased number of images in training 
        "learning_rate": 0.01, # Decreased learning rate 
        "epochs": 5, # Fixed number of iterations based on Kaggle 
        "num_workers": 2,  # Decreased number of parallel data loading workers  
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data", # Make sure this directory exists
        "ood_dir": "./data/ood-test",  
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }
    
    # Define GPU usage to 1 device to avoid SCC run abortion 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

############################################################################
# Data Loading
############################################################################
    # Load CIFAR-100 dataset from data transformation 
    trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    
    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

############################################################################
# Instantiate model and move to target device
############################################################################
   # Initialize model on target device 
    model = SimpleCNN().to(CONFIG["device"])

############################################################################
# Loss Function, Optimizer and optional learning rate scheduler
############################################################################
    # Initialize loss function 
    criterion = nn.CrossEntropyLoss()
    # Initialize SGD optimizer 
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9)
    # Initialize LR scheduler 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # To consistently decrease learning rate 
    
    # Initialize wandb
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model) # Watch the model gradients

################################################################################
# Training Loop 
################################################################################
    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        # Training loop 
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        scheduler.step()
        
        # Evaluation on testing 
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(testloader, desc="[Validation]"):
                inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        # Log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(trainloader),
            "train_acc": train_acc,
            "val_loss": test_loss / len(testloader),
            "val_acc": val_acc,
        })
        
        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
    
################################################################################
# Evaluation 
################################################################################
    # Evaluation on Clean CIFAR-100 Test Set
    predictions, clean_accuracy = evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Final CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    
    # Evaluation on OOD 
    all_predictions = evaluate_ood_test(model, CONFIG)
    
    # Create Submission File (OOD) 
    submission_df_ood = create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("OOD results saved successfully.")
    
if __name__ == '__main__':
    main()
