import os
import numpy as np  # type: ignore
import torch  # type: ignore
import imageio.v2 as imio  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from model import CNN

# Instantiate and load the pre-trained model
model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("q1_model.pt", map_location=device))
model.eval()

# Save visualization of the filters
# Extract the weights from the first convolutional layer
conv_weights = model.conv1.weight.data.cpu().numpy()  # Shape: (16, 3, 7, 7)

# Save each filter
os.makedirs("q2_filters", exist_ok=True)
for i in range(conv_weights.shape[0]):
    # Get the i-th filter (shape: 3 x 7 x 7)
    filter = conv_weights[i]
    
    # Normalize the filter to [0, 255]
    filter_min, filter_max = filter.min(), filter.max()
    filter = 255 * (filter - filter_min) / (filter_max - filter_min)
    filter = filter.astype(np.uint8)
    
    # Save the filter as an image
    for j in range(filter.shape[0]):
        imio.imwrite(f"q2_filters/filter_{i}_channel_{j}.png", filter[j])

# Visualize the filters
# Normalize the weights to [0, 1] for visualization with matplotlib imshow
conv_weights = (conv_weights - conv_weights.min()) / (conv_weights.max() - conv_weights.min())

# Filters and channels
num_filters = conv_weights.shape[0]
num_channels = conv_weights.shape[1]

# Plot the filters
fig, axes = plt.subplots(num_filters, num_channels, figsize=(8, 16))
for i in range(num_filters):
    for j in range(num_channels):
        filter = conv_weights[i, j]
        
        axes[i, j].imshow(filter, cmap='gray')
        axes[i, j].axis('off')
        
        if i == 0:
            axes[i, j].set_title(f'Channel {j+1}')
plt.tight_layout()
plt.savefig('q2_filters/all_filters.png', bbox_inches='tight')
plt.show()
