import torch
import matplotlib.pyplot as plt
from src.models import SimpleUNet, DDPMScheduler
import os
import argparse
import numpy as np

def get_model_config(dataset_name):
    if dataset_name == "mnist":
        return {
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 64,
            "image_size": 28,
            "num_channels": 1
        }
    elif dataset_name == "cifar10":
        return {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 128,
            "image_size": 32,
            "num_channels": 3
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def find_latest_model(dataset_name):
    models_dir = f"models_{dataset_name}"
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"No models directory found: {models_dir}")
    
    final_model = os.path.join(models_dir, "final_model.pth")
    if os.path.exists(final_model):
        return final_model
    
    epoch_models = [f for f in os.listdir(models_dir) if f.startswith("model_epoch_") and f.endswith(".pth")]
    if not epoch_models:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    latest_epoch = max([int(f.split("_")[-1].split(".")[0]) for f in epoch_models])
    return os.path.join(models_dir, f"model_epoch_{latest_epoch}.pth")

def generate_unconditional_images(dataset_name, num_images=8, num_steps=1000, model_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Generating {num_images} unconditional {dataset_name.upper()} images...")
    
    if model_path is None:
        model_path = find_latest_model(dataset_name)
    
    print(f"Using model: {model_path}")
    
    config = get_model_config(dataset_name)
    model = SimpleUNet(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        base_channels=config["base_channels"]
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scheduler = DDPMScheduler(num_timesteps=num_steps, device=device)
    
    image_size = config["image_size"]
    num_channels = config["num_channels"]
    
    with torch.no_grad():
        images = torch.randn(num_images, num_channels, image_size, image_size).to(device)
        
        for t in range(num_steps - 1, -1, -1):
            timesteps = torch.full((num_images,), t, device=device)
            
            predicted_noise = model(images, timesteps)
            images = scheduler.step(predicted_noise, t, images)
            
            if t % 100 == 0:
                print(f"Step {t}")
    
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    rows = 2
    cols = num_images // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    for i in range(num_images):
        row = i // cols
        col = i % cols
        
        if num_channels == 1:
            axes[row, col].imshow(images[i].cpu().squeeze(), cmap='gray')
        else:
            axes[row, col].imshow(images[i].cpu().permute(1, 2, 0))
        
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Generated {i+1}')
    
    plt.tight_layout()
    filename = f'unconditional_{dataset_name}_generated.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Generated images saved as '{filename}'")

def main():
    parser = argparse.ArgumentParser(description="Generate unconditional images using trained diffusion model")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist",
                        help="Dataset model was trained on")
    parser.add_argument("--num_images", type=int, default=8, help="Number of images to generate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of denoising steps")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model file")
    parser.add_argument("--quick", action="store_true", help="Quick generation with fewer steps")
    
    args = parser.parse_args()
    
    if args.quick:
        args.num_steps = 50
        print("Quick mode: Using 50 denoising steps")
    
    generate_unconditional_images(
        dataset_name=args.dataset,
        num_images=args.num_images,
        num_steps=args.num_steps,
        model_path=args.model_path
    )

if __name__ == "__main__":
    main()