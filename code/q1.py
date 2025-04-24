import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from data import create_dataloaders
from model import CNN

import os
from glob import glob
from model import CNN  # type: ignore
import torch  # type: ignore
from PIL import Image  # type: ignore
from torchvision.transforms import v2  # type: ignore

# For reproducibility
torch.manual_seed(123)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get dataloaders
train_loader, val_loader = create_dataloaders()

# Instantiate model
model = CNN().to(device)

# Initialize weights
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu', generator=None)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
model.apply(init_weights)

# Set training parameters
num_epochs = 60
lr = 1e-3
eta_min = 1e-5
weight_decay = 1e-3

# Setup cross-entropy loss function
loss_fn = nn.CrossEntropyLoss()

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Warm-up phase
warmup_epochs = 10
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)

# CosineAnnealingLR after warm-up
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=eta_min)

# Combine schedulers
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

# For plotting
step = 0
train_step_list, train_loss_list, train_accuracy_list = [], [], []
val_step_list, val_loss_list, val_accuracy_list = [], [], []

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # Move images and labels to device
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass through the model
        outputs = model(images)
        
        # Calculate the loss
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        train_loss_list.append(loss.item())
        train_step_list.append(step)
        
        step += 1
        
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
    # Learning rate scheduler
    scheduler.step()
    
    # Calculate validation loss and accuracy
    model.eval()
    with torch.no_grad():
        # Compute validation loss and accuracy
        correct, total = 0, 0
        avg_loss = 0.
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            avg_loss += loss.item() * labels.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        val_accuracy = correct / total * 100
        avg_loss /= total
        
        val_loss_list.append(avg_loss)
        val_accuracy_list.append(val_accuracy)
        val_step_list.append(step)
        
        # Calculate training accuracy
        train_correct, train_total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_accuracy = train_correct / train_total * 100
        train_accuracy_list.append(train_accuracy)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val acc: {val_accuracy:.2f}%,", 
                f"Train acc: {train_accuracy:.2f}%")
        
        if val_accuracy == max(val_accuracy_list):
            torch.save(model.state_dict(), "q1_model.pt")
        
        # Plot loss and accuracy
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(train_step_list, train_loss_list, label="Train")
        axs[0].plot(val_step_list, val_loss_list, label="Val")
        axs[0].set_yscale("log")

        axs[1].plot(val_step_list, train_accuracy_list, label="Train")
        axs[1].plot(val_step_list, val_accuracy_list, label="Val")

        axs[0].set_title("Loss")
        axs[1].set_title("Accuracy")
        
        for ax in axs:
            ax.legend()
            ax.grid()
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
        
        plt.tight_layout()
        plt.savefig(f"q1_plots.png", dpi=300)
        plt.clf()
        plt.close()

# Test the model
model = CNN()
model.load_state_dict(torch.load("q1_model.pt", weights_only=True))
model.to(device)
model.eval()

# Test transformations (similar to validation transformation)
test_tf = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.438, 0.435, 0.422], std=[0.228, 0.225, 0.231])
])

# Load the test images
test_images = sorted(glob.glob("../custom_image_dataset/test_unlabeled/*.png"))

# Test prediction
test_write = open("q1_test.txt", "w")
for imgfile in test_images:
    filename = os.path.basename(imgfile)
    img = Image.open(imgfile)
    img = test_tf(img)
    img = img.unsqueeze(0).to(device)
    
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    
    test_write.write(f"{filename},{predicted.item()}\n")
test_write.close()