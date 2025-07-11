import torch
import matplotlib.pyplot as plt
from src.models import ConditionedUNet, DDPMScheduler, BertTextTokenizer, SimpleTokenizer
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

def find_latest_model(dataset_name, tokenizer_type):
    models_dir = f"models_{dataset_name}_conditioned_{tokenizer_type}"
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"No models directory found: {models_dir}")
    
    final_model = os.path.join(models_dir, "final_conditioned_model.pth")
    final_tokenizer = os.path.join(models_dir, "final_tokenizer.pth")
    
    if os.path.exists(final_model) and os.path.exists(final_tokenizer):
        return final_model, final_tokenizer
    
    epoch_models = [f for f in os.listdir(models_dir) if f.startswith("conditioned_model_epoch_") and f.endswith(".pth")]
    if not epoch_models:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    latest_epoch = max([int(f.split("_")[-1].split(".")[0]) for f in epoch_models])
    model_path = os.path.join(models_dir, f"conditioned_model_epoch_{latest_epoch}.pth")
    tokenizer_path = os.path.join(models_dir, f"tokenizer_epoch_{latest_epoch}.pth")
    
    return model_path, tokenizer_path

def load_tokenizer(tokenizer_path, tokenizer_type, dataset_name):
    if tokenizer_type == "bert":
        if os.path.exists(tokenizer_path):
            try:
                return torch.load(tokenizer_path, map_location="cpu", weights_only=False)
            except:
                print("Could not load saved tokenizer, using default BERT tokenizer")
                return BertTextTokenizer()
        else:
            print("Tokenizer file not found, using default BERT tokenizer")
            return BertTextTokenizer()
    
    elif tokenizer_type == "simple":
        if os.path.exists(tokenizer_path):
            try:
                return torch.load(tokenizer_path, map_location="cpu", weights_only=False)
            except:
                print("Could not load saved tokenizer, using default Simple tokenizer")
                return SimpleTokenizer(dataset=dataset_name)
        else:
            print("Tokenizer file not found, using default Simple tokenizer")
            return SimpleTokenizer(dataset=dataset_name)
    
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

def get_sample_prompts(dataset_name, tokenizer_type):
    if dataset_name == "mnist":
        if tokenizer_type == "simple":
            return [
                "handwritten digit zero",
                "handwritten digit one", 
                "handwritten digit five",
                "digit 7"
            ]
        else:  # bert
            return [
                "handwritten digit zero",
                "the number 1",
                "handwritten digit five",
                "digit seven"
            ]
    
    elif dataset_name == "cifar10":
        if tokenizer_type == "simple":
            return [
                "airplane",
                "car",
                "bird",
                "cat"
            ]
        else:  # bert
            return [
                "a photo of an airplane",
                "a red car on road",
                "a colorful bird flying",
                "a cute cat sitting"
            ]
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def generate_conditioned_images(dataset_name, tokenizer_type, prompts=None, num_images=8, 
                               num_steps=1000, model_path=None, tokenizer_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Generating {num_images} conditioned {dataset_name.upper()} images...")
    print(f"Using {tokenizer_type.upper()} tokenizer")
    
    if model_path is None or tokenizer_path is None:
        model_path, tokenizer_path = find_latest_model(dataset_name, tokenizer_type)
    
    print(f"Using model: {model_path}")
    print(f"Using tokenizer: {tokenizer_path}")
    
    config = get_model_config(dataset_name)
    tokenizer = load_tokenizer(tokenizer_path, tokenizer_type, dataset_name)
    
    model = ConditionedUNet(tokenizer=tokenizer, **{k: v for k, v in config.items() if k in ["in_channels", "out_channels", "base_channels"]}).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scheduler = DDPMScheduler(num_timesteps=num_steps, device=device)
    
    if prompts is None:
        prompts = get_sample_prompts(dataset_name, tokenizer_type)
    
    if len(prompts) == 1:
        prompts = prompts * num_images
    elif len(prompts) < num_images:
        prompts = (prompts * ((num_images // len(prompts)) + 1))[:num_images]
    else:
        prompts = prompts[:num_images]
    
    print(f"Using prompts: {prompts}")
    
    image_size = config["image_size"]
    num_channels = config["num_channels"]
    
    text_tokens = tokenizer.encode_batch(prompts).to(device)
    
    with torch.no_grad():
        images = torch.randn(num_images, num_channels, image_size, image_size).to(device)
        
        for t in range(num_steps - 1, -1, -1):
            timesteps = torch.full((num_images,), t, device=device)
            
            predicted_noise = model(images, timesteps, text_tokens)
            images = scheduler.step(predicted_noise, t, images)
            
            if t % 100 == 0:
                print(f"Step {t}")
    
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    rows = 2
    cols = num_images // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    for i in range(num_images):
        row = i // cols
        col = i % cols
        
        if num_channels == 1:
            axes[row, col].imshow(images[i].cpu().squeeze(), cmap='gray')
        else:
            axes[row, col].imshow(images[i].cpu().permute(1, 2, 0))
        
        axes[row, col].axis('off')
        axes[row, col].set_title(f'"{prompts[i]}"', fontsize=8)
    
    plt.tight_layout()
    filename = f'conditioned_{dataset_name}_{tokenizer_type}_generated.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Generated images saved as '{filename}'")

def main():
    parser = argparse.ArgumentParser(description="Generate conditioned images using trained diffusion model")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist",
                        help="Dataset model was trained on")
    parser.add_argument("--tokenizer", type=str, choices=["bert", "simple"], default="bert",
                        help="Tokenizer type used during training")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--num_images", type=int, default=8, help="Number of images to generate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of denoising steps")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model file")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer file")
    parser.add_argument("--quick", action="store_true", help="Quick generation with fewer steps")
    parser.add_argument("--demo", action="store_true", help="Generate demo images with sample prompts")
    
    args = parser.parse_args()
    
    if args.quick:
        args.num_steps = 50
        print("Quick mode: Using 50 denoising steps")
    
    prompts = None
    if args.prompt:
        prompts = [args.prompt]
    elif args.demo:
        prompts = get_sample_prompts(args.dataset, args.tokenizer)
        print("Demo mode: Using sample prompts")
    
    generate_conditioned_images(
        dataset_name=args.dataset,
        tokenizer_type=args.tokenizer,
        prompts=prompts,
        num_images=args.num_images,
        num_steps=args.num_steps,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )

if __name__ == "__main__":
    main()