import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import wandb
from tqdm.auto import tqdm

# Import eval_cifar100 and eval_ood to ensure tests run correctly on SCC
from eval_cifar100 import evaluate_cifar100_test
from eval_ood import evaluate_ood_test, create_ood_df

################################################################################
# Model Definition - Pretrained CNN using ResNet50
################################################################################
class PretrainedCNN(nn.Module):
    # ResNet50 model 
    def __init__(self):
        super(PretrainedCNN, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        
        # Unfreeze latter half of layers for fine-tuning
        unfreeze_layers = len(list(self.model.parameters())) // 2
        for param in list(self.model.parameters())[:-unfreeze_layers]:  
            param.requires_grad = False
        
        # Replace final layer for CIFAR-100
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 100)  # 100 classes in CIFAR-100
        
    # Forward pass 
    def forward(self, x):
        return self.model(x)

################################################################################
# Data Transformation - Modified 
################################################################################
transform_train = transforms.Compose([
    # Random changes for variation 
    transforms.ToTensor(),  # Convert images to tensor first
    transforms.RandomHorizontalFlip(), # Flip images to horizontal
    transforms.RandomRotation(20), # Rotate more images 
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # Change colors 
    transforms.RandomCrop(32, padding=4), # Crop images with padding 
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)), # Erase smaller areas 
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), # Normalize based on CIFAR-100 statistics
])

transform_test = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensor first
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # Normalize based on CIFAR-100 statistics
])

################################################################################
# Configuration Dictionary - Modified 
################################################################################
def main():
    CONFIG = {
        "model": "PretrainedCNN",
        "batch_size": 64, # Increased number of images in training
        "learning_rate": 0.003, # Increased learning rate from 0.001 to 0.003 
        "epochs": 50,  # Fixed number of iterations based on Kaggle 
        "num_workers": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data", # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
        "weight_decay": 2e-4, # Added weight decay to increase regularization and decrease overfitting 
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
    model = PretrainedCNN().to(CONFIG["device"])

############################################################################
# Loss Function, Optimizer and optional learning rate scheduler
############################################################################
    # Initialize loss function 
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Add smoothing to improve generalization
    # Initialize AdamW optimizer 
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]) # To regularize better
    # Initialize LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"]) # To gradually decrease learning rate
    
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
            torch.save(model.state_dict(), "best_model_pretrained.pth") # Update .pth name 
    
################################################################################
# Evaluation 
################################################################################
    # Evaluation on Clean CIFAR-100 Test Set
    model.load_state_dict(torch.load("best_model_pretrained.pth"))
    predictions, clean_accuracy = evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Final CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    
    # Evaluation on OOD 
    all_predictions = evaluate_ood_test(model, CONFIG)
    
    # Create Submission File (OOD) 
    submission_df_ood = create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood_pretrained.csv", index=False)
    print("OOD results saved successfully.")
    
if __name__ == '__main__':
    main()
