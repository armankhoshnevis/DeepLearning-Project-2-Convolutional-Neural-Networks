import torch  # type: ignore
import torch.nn as nn  # type: ignore

torch.manual_seed(123)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Convolutional layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(48)
        
        # Convolutional layer 4
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Convolutional layer 5
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, padding=1, stride=1)
        
        # Dropout Layer
        self.dropout = nn.Dropout(p=0.1)
        
        # ReLU Activation
        self.relu = nn.ReLU()
        
        # Max and Adaptive Average Pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear Layer
        self.fc = nn.Linear(80, 10)
    
    def forward(self, x, intermediate_outputs=False):
        # Conv1 -> BN -> ReLU
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        
        # Conv2 -> BN -> ReLU -> MaxPool
        conv2_out = self.relu(self.bn2(self.conv2(conv1_out)))
        maxpool1_out = self.maxpool(conv2_out)
        
        # Conv3 -> BN -> ReLU -> MaxPool
        conv3_out = self.relu(self.bn3(self.conv3(maxpool1_out)))
        conv3_out = self.dropout(conv3_out)
        maxpool2_out = self.maxpool(conv3_out)
        
        # Conv4 -> BN -> ReLU -> MaxPool
        conv4_out = self.relu(self.bn4(self.conv4(maxpool2_out)))
        conv4_out = self.dropout(conv4_out)
        maxpool3_out = self.maxpool(conv4_out)
        
        # Conv5 -> AvgPool
        conv5_out = self.conv5(maxpool3_out)
        conv5_out = self.dropout(conv5_out)
        avgpool_out = self.avgpool(conv5_out)
        
        # Flatten
        flattened = torch.flatten(avgpool_out, start_dim=1)
        
        # Linear Layer
        final_out = self.fc(flattened)
        
        # Return intermediate outputs if requested
        if intermediate_outputs:
            return final_out, [conv1_out, conv2_out, conv3_out, conv4_out, conv5_out]
        else:
            return final_out
