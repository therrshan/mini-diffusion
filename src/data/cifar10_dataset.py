import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional
import os

class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir: str = "data", train: bool = True, transform=None):
        self.data_dir = data_dir
        self.train = train
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
            
        self.dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=self.train,
            download=True,
            transform=self.transform
        )
        
        self.class_to_text = {
            0: "airplane",
            1: "automobile", 
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck"
        }
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        text = self.class_to_text[label]
        return image, text, label

def get_cifar10_dataloader(data_dir: str = "data", batch_size: int = 64, 
                          train: bool = True, num_workers: int = 4) -> DataLoader:
    dataset = CIFAR10Dataset(data_dir=data_dir, train=train)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )