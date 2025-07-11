import torch
import matplotlib.pyplot as plt
from src.models import SimpleUNet, DDPMScheduler
import os

def generate_images(model_path="models/model_epoch_20.pth", num_images=8, num_steps=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = SimpleUNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scheduler = DDPMScheduler(num_timesteps=num_steps, device=device)
    
    print(f"Generating {num_images} images...")
    
    with torch.no_grad():
        images = torch.randn(num_images, 1, 28, 28).to(device)
        
        for t in range(num_steps - 1, -1, -1):
            timesteps = torch.full((num_images,), t, device=device)
            
            predicted_noise = model(images, timesteps)
            images = scheduler.step(predicted_noise, t, images)
            
            if t % 100 == 0:
                print(f"Step {t}")
    
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(num_images):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Generated {i+1}')
    
    plt.tight_layout()
    plt.savefig('generated_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Generated images saved as 'generated_images.png'")

def quick_generate(model_path="models/final_model.pth", num_images=4, num_steps=50):
    print("Quick generation (fewer steps, faster)...")
    generate_images(model_path, num_images, num_steps)

if __name__ == "__main__":
    if os.path.exists("models/model_epoch_20.pth"):
        generate_images()
    elif any(f.startswith("model_epoch_") for f in os.listdir("models") if f.endswith(".pth")):
        latest_epoch = max([int(f.split("_")[-1].split(".")[0]) for f in os.listdir("models") if f.startswith("model_epoch_")])
        model_path = f"models/model_epoch_{latest_epoch}.pth"
        print(f"Using checkpoint: {model_path}")
        generate_images(model_path)
    else:
        print("No trained model found! Train the model first with: python train.py")