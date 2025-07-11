import torch
import matplotlib.pyplot as plt
from src.models import ConditionedUNet, DDPMScheduler, SimpleTokenizer
import os

def generate_conditioned_images(prompt="handwritten digit eight", num_images=8, 
                               model_path="models_conditioned/final_conditioned_model.pth",
                               tokenizer_path="models_conditioned/final_tokenizer.pth",
                               num_steps=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Generating {num_images} images for prompt: '{prompt}'")
    
    if os.path.exists(tokenizer_path):
        try:
            tokenizer = torch.load(tokenizer_path, map_location=device, weights_only=False)
        except:
            print("Could not load saved tokenizer, using default")
            tokenizer = SimpleTokenizer()
    else:
        tokenizer = SimpleTokenizer()
        print("Using default tokenizer (tokenizer file not found)")
    
    model = ConditionedUNet(
        in_channels=1, 
        out_channels=1, 
        vocab_size=tokenizer.vocab_size
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scheduler = DDPMScheduler(num_timesteps=num_steps, device=device)
    
    text_tokens = tokenizer.encode_batch([prompt] * num_images).to(device)
    
    with torch.no_grad():
        images = torch.randn(num_images, 1, 28, 28).to(device)
        
        for t in range(num_steps - 1, -1, -1):
            timesteps = torch.full((num_images,), t, device=device)
            
            predicted_noise = model(images, timesteps, text_tokens)
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
        axes[row, col].set_title(f'"{prompt}"')
    
    plt.tight_layout()
    filename = f'conditioned_images_{prompt.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Generated images saved as '{filename}'")

def test_multiple_prompts():
    prompts = [
        "handwritten digit zero",
        "handwritten digit one", 
        "handwritten digit five",
        "digit 7",
        "nine"
    ]
    
    for prompt in prompts:
        print(f"\n--- Testing prompt: '{prompt}' ---")
        generate_conditioned_images(prompt, num_images=4, num_steps=200)

def quick_test(prompt="handwritten digit five"):
    print(f"Quick test for: '{prompt}'")
    generate_conditioned_images(prompt, num_images=4, num_steps=50)

if __name__ == "__main__":
    if os.path.exists("models_conditioned/final_conditioned_model.pth"):
        generate_conditioned_images()
    elif any(f.startswith("conditioned_model_epoch_") for f in os.listdir("models_conditioned") if f.endswith(".pth")):
        latest_epoch = max([int(f.split("_")[-1].split(".")[0]) for f in os.listdir("models_conditioned") if f.startswith("conditioned_model_epoch_")])
        model_path = f"models_conditioned/conditioned_model_epoch_{latest_epoch}.pth"
        tokenizer_path = f"models_conditioned/tokenizer_epoch_{latest_epoch}.pth"
        print(f"Using checkpoint: {model_path}")
        generate_conditioned_images(model_path=model_path, tokenizer_path=tokenizer_path)
    else:
        print("No trained conditioned model found! Train first with: python train_conditioned.py")