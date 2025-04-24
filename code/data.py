import torch  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from torchvision.datasets import ImageFolder  # type: ignore
from torchvision.transforms import v2  # type: ignore

torch.manual_seed(123)  # For reproducibility

def create_dataloaders():
    train_tf = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),  # Flip the image horizontally and randomly 
        v2.ToImage(),  # Convert the tensor to image (does not scale)
        v2.ToDtype(torch.float32, scale=True),  # Convert the tensor to float32 and scale to [0, 1]
        v2.Normalize(mean=[0.438, 0.435, 0.422], std=[0.228, 0.225, 0.231]),  # Normalize the image
        v2.GaussianNoise(sigma=0.05, clip=False),  # Add Gaussian noise
        v2.RandomErasing(p=0.3, scale=(0.02, 0.05)),  # Random erasing part of the image
    ])
    
    val_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.438, 0.435, 0.422], std=[0.228, 0.225, 0.231])
    ])
    
    # Load the train and validation datasets
    train_dataset = ImageFolder(root='../custom_image_dataset/train', transform=train_tf)
    val_dataset = ImageFolder(root='../custom_image_dataset/val', transform=val_tf)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    return train_loader, val_loader
