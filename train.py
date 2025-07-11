import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from src.models import SimpleUNet, DDPMScheduler

def get_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def train_model(num_epochs=50, batch_size=64, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataloader = get_dataloader(batch_size)
    
    model = SimpleUNet(in_channels=1, out_channels=1).to(device)
    scheduler = DDPMScheduler(num_timesteps=1000, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images = images.to(device)
            batch_size = images.shape[0]
            
            noise = torch.randn_like(images)
            timesteps = scheduler.sample_timesteps(batch_size)
            noisy_images = scheduler.add_noise(images, noise, timesteps)
            
            predicted_noise = model(noisy_images, timesteps)
            loss = criterion(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/model_epoch_{epoch+1}.pth")
    
    torch.save(model.state_dict(), "models/final_model.pth")
    print("Training complete!")

if __name__ == "__main__":
    train_model()