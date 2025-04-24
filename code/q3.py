import torch  # type: ignore
from torch import linalg as LA  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import os

from data import create_dataloaders
from model import CNN

# Load the model
model = CNN()
train_loader, test_loader = create_dataloaders()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the weights
model.load_state_dict(torch.load("q1_model.pt", weights_only=True))
model.to(device)

os.makedirs("q3_filters", exist_ok=True)

# Number of output channels in each layer
num_fs = [16, 32, 48, 64, 80]

# Initialize tensors to store norms and labels
norms = torch.zeros(len(test_loader.dataset), sum(num_fs))
all_labels = torch.zeros(len(test_loader.dataset))

model.eval()

step = 0
for images, labels in test_loader:
    # Move to device
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass with intermediate outputs
    _, [conv1_out, conv2_out, conv3_out, conv4_out, conv5_out] = model(images, intermediate_outputs=True)
    
    # Compute channel-wise norms
    x1 = LA.vector_norm(conv1_out, ord=2, dim=(2, 3))
    x2 = LA.vector_norm(conv2_out, ord=2, dim=(2, 3))
    x3 = LA.vector_norm(conv3_out, ord=2, dim=(2, 3))
    x4 = LA.vector_norm(conv4_out, ord=2, dim=(2, 3))
    x5 = LA.vector_norm(conv5_out, ord=2, dim=(2, 3))
    
    # Store the norms in the `norms` tensor
    f_idx = 0
    norms[step:step+images.size(0), f_idx:f_idx+num_fs[0]] = x1.detach().cpu()
    f_idx += num_fs[0]
    norms[step:step+images.size(0), f_idx:f_idx+num_fs[1]] = x2.detach().cpu()
    f_idx += num_fs[1]
    norms[step:step+images.size(0), f_idx:f_idx+num_fs[2]] = x3.detach().cpu()
    f_idx += num_fs[2]
    norms[step:step+images.size(0), f_idx:f_idx+num_fs[3]] = x4.detach().cpu()
    f_idx += num_fs[3]
    norms[step:step+images.size(0), f_idx:f_idx+num_fs[4]] = x5.detach().cpu()
    
    # Save the labels
    all_labels[step:step+images.size(0)] = labels
    
    step += images.size(0)

# Class-wise average activation
labelnames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

classwise_score_avg = torch.zeros(10, sum(num_fs))
for l in range(10):
    classwise_score_avg[l] = norms[all_labels == l].mean(dim=0)

# Plot class-wise average activation for each filter
start = 0
for layer_idx, num_f in enumerate(num_fs):
    os.makedirs(f"q3_filters/classwise_avg_{layer_idx}", exist_ok=True)

    for f_idx in range(num_f):
        fig, ax = plt.subplots()
        data = classwise_score_avg[:, start+f_idx]
        data /= data.max()  # Normalize by the maximum value
        ax.bar(labelnames, data)
        ax.set_title(f"Filter {f_idx}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"q3_filters/classwise_avg_{layer_idx}/filter_{f_idx}.png")
        plt.close()

    start += num_f
