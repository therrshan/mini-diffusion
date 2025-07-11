import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
from src.models import SimpleUNet, DDPMScheduler

def get_dataloader(dataset_name, batch_size=64, data_dir="data"):
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use 'mnist' or 'cifar10'.")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def get_model_config(dataset_name):
    if dataset_name == "mnist":
        return {
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 64
        }
    elif dataset_name == "cifar10":
        return {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 128
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def plot_training_curves(epoch_losses, batch_losses, save_dir, dataset_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epoch_losses)
    ax1.set_title(f'{dataset_name.upper()} Training Loss (Per Epoch)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss')
    ax1.grid(True)
    
    ax2.plot(batch_losses, alpha=0.7)
    ax2.set_title(f'{dataset_name.upper()} Training Loss (Per Batch)')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    epochs = len(epoch_losses)
    batches_per_epoch = len(batch_losses) // epochs if epochs > 0 else 1
    epoch_ticks = [i * batches_per_epoch for i in range(epochs)]
    for tick in epoch_ticks:
        ax2.axvline(x=tick, color='red', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_dir}/training_curves.png")

def save_training_stats(epoch_losses, batch_losses, save_dir):
    stats = {
        'epoch_losses': epoch_losses,
        'batch_losses': batch_losses,
        'final_loss': epoch_losses[-1] if epoch_losses else 0,
        'best_loss': min(epoch_losses) if epoch_losses else 0
    }
    torch.save(stats, f'{save_dir}/training_stats.pth')
    print(f"Training statistics saved to {save_dir}/training_stats.pth")

def train_model(dataset_name, num_epochs=50, batch_size=64, lr=1e-4, data_dir="data"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training on {dataset_name.upper()} dataset")
    
    dataloader = get_dataloader(dataset_name, batch_size, data_dir)
    model_config = get_model_config(dataset_name)
    
    model = SimpleUNet(**model_config).to(device)
    scheduler = DDPMScheduler(num_timesteps=1000, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Starting training for {num_epochs} epochs...")
    
    save_dir = f"models_{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    epoch_losses = []
    batch_losses = []
    
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
            batch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}")
        
        if (epoch + 1) % 5 == 0:
            plot_training_curves(epoch_losses, batch_losses, save_dir, dataset_name)
        
        if (epoch + 1) % 10 == 0:
            model_path = f"{save_dir}/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Saved checkpoint: {model_path}")
    
    final_model_path = f"{save_dir}/final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    
    plot_training_curves(epoch_losses, batch_losses, save_dir, dataset_name)
    save_training_stats(epoch_losses, batch_losses, save_dir)
    
    print(f"Training complete! Final model saved: {final_model_path}")
    print(f"Final loss: {epoch_losses[-1]:.6f}")
    print(f"Best loss: {min(epoch_losses):.6f}")

def main():
    parser = argparse.ArgumentParser(description="Train diffusion model on MNIST or CIFAR-10")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist", 
                        help="Dataset to train on (mnist or cifar10)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    
    args = parser.parse_args()
    
    if args.dataset == "cifar10" and args.batch_size > 32:
        print("Warning: Large batch size for CIFAR-10 may cause memory issues. Consider using --batch_size 32")
    
    train_model(
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir
    )

if __name__ == "__main__":
    main()