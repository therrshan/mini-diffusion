import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional
import os

class MNISTDataset(Dataset):
    def __init__(self, data_dir: str = "data", train: bool = True, transform=None):
        self.data_dir = data_dir
        self.train = train
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transform
            
        self.dataset = datasets.MNIST(
            root=self.data_dir,
            train=self.train,
            download=True,
            transform=self.transform
        )
        
        self.digit_to_text = {
            0: "handwritten digit zero",
            1: "handwritten digit one", 
            2: "handwritten digit two",
            3: "handwritten digit three",
            4: "handwritten digit four",
            5: "handwritten digit five",
            6: "handwritten digit six",
            7: "handwritten digit seven",
            8: "handwritten digit eight",
            9: "handwritten digit nine"
        }
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        text = self.digit_to_text[label]
        return image, text, label

def get_mnist_dataloader(data_dir: str = "data", batch_size: int = 64, 
                        train: bool = True, num_workers: int = 4) -> DataLoader:
    dataset = MNISTDataset(data_dir=data_dir, train=train)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )