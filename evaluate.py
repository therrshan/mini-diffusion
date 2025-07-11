import argparse
import json
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from src.models import SimpleUNet, ConditionedUNet, DDPMScheduler, BertTextTokenizer, SimpleTokenizer
import time

def calculate_fid_score(real_images, generated_images):
    """Calculate Frechet Inception Distance between real and generated images"""
    def get_activations(images):
        activations = []
        for img in images:
            if len(img.shape) == 4:
                img = img.squeeze(0)
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            
            img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            
            features = np.mean(img_np.reshape(-1, img_np.shape[-1]), axis=0)
            activations.append(features)
        
        return np.array(activations)
    
    try:
        real_acts = get_activations(real_images)
        gen_acts = get_activations(generated_images)
        
        mu1, sigma1 = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
        mu2, sigma2 = gen_acts.mean(axis=0), np.cov(gen_acts, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2)**2.0)
        
        if sigma1.ndim == 0:
            sigma1 = np.array([[sigma1]])
        if sigma2.ndim == 0:
            sigma2 = np.array([[sigma2]])
        
        try:
            covmean = sqrtm(sigma1.dot(sigma2))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
        except:
            covmean = 0
        
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return float(fid)
    except:
        return None

def calculate_inception_score(generated_images):
    """Calculate Inception Score for generated images"""
    try:
        scores = []
        for img in generated_images:
            if len(img.shape) == 4:
                img = img.squeeze(0)
            
            img_np = img.detach().cpu().numpy()
            if img_np.shape[0] == 1:
                diversity = np.std(img_np)
            else:
                diversity = np.mean([np.std(img_np[c]) for c in range(img_np.shape[0])])
            
            scores.append(diversity)
        
        return float(np.mean(scores))
    except:
        return None

def get_model_config(dataset_name):
    """Get model configuration for dataset"""
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

def load_tokenizer(tokenizer_path, tokenizer_type, dataset_name):
    """Load tokenizer from file or create default"""
    if tokenizer_type == "bert":
        if os.path.exists(tokenizer_path):
            try:
                return torch.load(tokenizer_path, map_location="cpu", weights_only=False)
            except:
                return BertTextTokenizer()
        else:
            return BertTextTokenizer()
    elif tokenizer_type == "simple":
        if os.path.exists(tokenizer_path):
            try:
                return torch.load(tokenizer_path, map_location="cpu", weights_only=False)
            except:
                return SimpleTokenizer(dataset=dataset_name)
        else:
            return SimpleTokenizer(dataset=dataset_name)

def get_evaluation_prompts(dataset_name, tokenizer_type):
    """Get prompts for evaluation"""
    if dataset_name == "mnist":
        return ["handwritten digit " + str(i) for i in range(10)]
    elif dataset_name == "cifar10":
        if tokenizer_type == "bert":
            return [
                "a photo of an airplane",
                "a red car on road",
                "a small bird flying",
                "a cute cat sitting",
                "a deer in forest"
            ]
        else:
            return ["airplane", "car", "bird", "cat", "deer"]

def get_real_images_sample(dataset_name, num_samples=50):
    """Get sample of real images for comparison"""
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=min(num_samples, 32), shuffle=True)
    
    real_images = []
    for images, _ in dataloader:
        real_images.extend(images)
        if len(real_images) >= num_samples:
            break
    
    return real_images[:num_samples]

def generate_sample_images(model, scheduler, config, device, num_samples=50, prompts=None, tokenizer=None):
    """Generate sample images using the model"""
    model.eval()
    generated_images = []
    
    with torch.no_grad():
        for i in range(0, num_samples, 4):
            batch_size = min(4, num_samples - i)
            images = torch.randn(batch_size, config["num_channels"], config["image_size"], config["image_size"]).to(device)
            
            if prompts and tokenizer:
                batch_prompts = prompts[:batch_size] if len(prompts) >= batch_size else prompts * (batch_size // len(prompts) + 1)
                batch_prompts = batch_prompts[:batch_size]
                text_tokens = tokenizer.encode_batch(batch_prompts).to(device)
                
                for t in range(49, -1, -1):
                    timesteps = torch.full((batch_size,), t, device=device)
                    predicted_noise = model(images, timesteps, text_tokens)
                    images = scheduler.step(predicted_noise, t, images)
            else:
                for t in range(49, -1, -1):
                    timesteps = torch.full((batch_size,), t, device=device)
                    predicted_noise = model(images, timesteps)
                    images = scheduler.step(predicted_noise, t, images)
            
            images = (images + 1) / 2
            images = torch.clamp(images, 0, 1)
            generated_images.extend(images.cpu())
    
    return generated_images

def load_training_stats(model_dir):
    """Load training statistics from model directory"""
    stats_path = os.path.join(model_dir, "training_stats.pth")
    if os.path.exists(stats_path):
        return torch.load(stats_path, map_location="cpu")
    return None

def save_sample_images(generated_samples, sample_prompts, output_dir, model_name):
    """Save generated sample images to files"""
    samples_dir = os.path.join(output_dir, f"{model_name}_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    for i, img in enumerate(generated_samples[:8]):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        
        if img.shape[0] == 1:
            ax.imshow(img.squeeze(), cmap='gray')
        else:
            ax.imshow(img.permute(1, 2, 0))
        
        ax.axis('off')
        
        if sample_prompts and i < len(sample_prompts):
            ax.set_title(sample_prompts[i], fontsize=10)
        
        filename = f"sample_{i+1}.png"
        if sample_prompts and i < len(sample_prompts):
            safe_prompt = "".join(c for c in sample_prompts[i][:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"sample_{i+1}_{safe_prompt.replace(' ', '_')}.png"
        
        plt.savefig(os.path.join(samples_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Sample images saved to: {samples_dir}")

def create_training_curves_plot(training_stats, output_dir, model_name):
    """Create and save training curves plot"""
    if not training_stats:
        print("⚠️ No training statistics available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epoch_losses = training_stats.get('epoch_losses', [])
    if epoch_losses:
        ax1.plot(epoch_losses, 'b-', linewidth=2)
        ax1.set_title('Training Loss (Per Epoch)', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Average Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
    
    batch_losses = training_stats.get('batch_losses', [])
    if batch_losses:
        step = max(1, len(batch_losses) // 2000)
        sampled_losses = batch_losses[::step]
        sampled_indices = list(range(0, len(batch_losses), step))
        
        ax2.plot(sampled_indices, sampled_losses, 'r-', alpha=0.7, linewidth=1)
        ax2.set_title('Training Loss (Per Batch - Sampled)', fontsize=14)
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    curves_path = os.path.join(output_dir, f"{model_name}_training_curves.png")
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {curves_path}")

def format_number(num):
    """Format large numbers in human readable format"""
    if num >= 1000000:
        return f"{num / 1000000:.2f}M"
    elif num >= 1000:
        return f"{num / 1000:.1f}K"
    return str(num)

def print_evaluation_results(results, model_name):
    """Print formatted evaluation results"""
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS FOR: {model_name.upper()}")
    print("="*60)
    
    print(f"Model Type: {results['model_type']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Model Size: {format_number(results['model_size'])} parameters")
    
    if 'tokenizer_vocab_size' in results:
        print(f"Tokenizer Vocab: {format_number(results['tokenizer_vocab_size'])} tokens")
    
    print("\nPERFORMANCE METRICS:")
    print("-" * 30)
    
    if results.get('fid_score') is not None:
        fid = results['fid_score']
        fid_quality = "Excellent" if fid < 20 else "Good" if fid < 50 else "Fair" if fid < 100 else "Poor"
        print(f"FID Score: {fid:.3f} ({fid_quality})")
    else:
        print("FID Score: Not available")
    
    if results.get('inception_score') is not None:
        is_score = results['inception_score']
        is_quality = "High" if is_score > 4 else "Medium" if is_score > 2 else "Low"
        print(f"Inception Score: {is_score:.3f} ({is_quality} diversity)")
    else:
        print("Inception Score: Not available")
    
    training_stats = results.get('training_stats')
    if training_stats:
        print("\nTRAINING STATISTICS:")
        print("-" * 30)
        print(f"Final Loss: {training_stats.get('final_loss', 'N/A'):.6f}")
        print(f"Best Loss: {training_stats.get('best_loss', 'N/A'):.6f}")
        print(f"Total Epochs: {len(training_stats.get('epoch_losses', []))}")
        print(f"Total Batches: {len(training_stats.get('batch_losses', []))}")
    
    print("\nEvaluation completed successfully!")
    print("="*60)

def evaluate_model(model_dir, dataset_name, model_type, tokenizer_type, device):
    """Main evaluation function"""
    try:
        config = get_model_config(dataset_name)
        
        if model_type == "unconditional":
            model_path = os.path.join(model_dir, "final_model.pth")
            if not os.path.exists(model_path):
                epoch_models = [f for f in os.listdir(model_dir) if f.startswith("model_epoch_")]
                if not epoch_models:
                    return None
                latest_epoch = max([int(f.split("_")[-1].split(".")[0]) for f in epoch_models])
                model_path = os.path.join(model_dir, f"model_epoch_{latest_epoch}.pth")
            
            model = SimpleUNet(**{k: v for k, v in config.items() if k in ["in_channels", "out_channels", "base_channels"]})
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            
            scheduler = DDPMScheduler(num_timesteps=50, device=device)
            training_stats = load_training_stats(model_dir)
            
            print("Generating sample images for evaluation...")
            generated_images = generate_sample_images(model, scheduler, config, device, num_samples=30)
            real_images = get_real_images_sample(dataset_name, num_samples=30)
            
            fid_score = calculate_fid_score(real_images, generated_images)
            is_score = calculate_inception_score(generated_images)
            model_size = sum(p.numel() for p in model.parameters())
            
            return {
                "model_type": "Unconditional",
                "dataset": dataset_name.upper(),
                "model_size": model_size,
                "training_stats": training_stats,
                "fid_score": fid_score,
                "inception_score": is_score,
                "generated_samples": generated_images[:8],
                "model_path": model_path
            }
            
        elif model_type == "conditioned":
            model_path = os.path.join(model_dir, "final_conditioned_model.pth")
            tokenizer_path = os.path.join(model_dir, "final_tokenizer.pth")
            
            if not os.path.exists(model_path):
                epoch_models = [f for f in os.listdir(model_dir) if f.startswith("conditioned_model_epoch_")]
                if not epoch_models:
                    return None
                latest_epoch = max([int(f.split("_")[-1].split(".")[0]) for f in epoch_models])
                model_path = os.path.join(model_dir, f"conditioned_model_epoch_{latest_epoch}.pth")
                tokenizer_path = os.path.join(model_dir, f"tokenizer_epoch_{latest_epoch}.pth")
            
            tokenizer = load_tokenizer(tokenizer_path, tokenizer_type, dataset_name)
            model = ConditionedUNet(tokenizer=tokenizer, **{k: v for k, v in config.items() if k in ["in_channels", "out_channels", "base_channels"]})
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            
            scheduler = DDPMScheduler(num_timesteps=50, device=device)
            training_stats = load_training_stats(model_dir)
            prompts = get_evaluation_prompts(dataset_name, tokenizer_type)
            
            print("Generating conditioned sample images for evaluation...")
            generated_images = generate_sample_images(model, scheduler, config, device, num_samples=20, prompts=prompts, tokenizer=tokenizer)
            real_images = get_real_images_sample(dataset_name, num_samples=20)
            
            fid_score = calculate_fid_score(real_images, generated_images)
            is_score = calculate_inception_score(generated_images)
            model_size = sum(p.numel() for p in model.parameters())
            
            return {
                "model_type": f"Text-Conditioned ({tokenizer_type.upper()})",
                "dataset": dataset_name.upper(),
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "model_size": model_size,
                "training_stats": training_stats,
                "fid_score": fid_score,
                "inception_score": is_score,
                "generated_samples": generated_images[:8],
                "sample_prompts": prompts[:8],
                "model_path": model_path
            }
    
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a specific diffusion model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_model.py --dataset mnist --model_type unconditional
  python evaluate_model.py --dataset cifar10 --model_type conditioned --tokenizer bert
  python evaluate_model.py --dataset mnist --model_type conditioned --tokenizer simple --output_dir my_results
        """
    )
    
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], required=True,
                        help="Dataset the model was trained on")
    parser.add_argument("--model_type", type=str, choices=["unconditional", "conditioned"], required=True,
                        help="Type of model to evaluate")
    parser.add_argument("--tokenizer", type=str, choices=["bert", "simple"], default="bert",
                        help="Tokenizer type (required for conditioned models)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--save_json", action="store_true",
                        help="Save results as JSON file")
    
    args = parser.parse_args()
    
    if args.model_type == "conditioned" and not args.tokenizer:
        parser.error("--tokenizer is required for conditioned models")
    
    if args.model_type == "unconditional":
        model_dir = f"models_{args.dataset}"
        model_name = f"{args.dataset}_unconditional"
    else:
        model_dir = f"models_{args.dataset}_conditioned_{args.tokenizer}"
        model_name = f"{args.dataset}_conditioned_{args.tokenizer}"
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found!")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting evaluation of {model_name}")
    print(f"Model directory: {model_dir}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    print("\n⏳ Running evaluation...")
    start_time = time.time()
    
    tokenizer_type = args.tokenizer if args.model_type == "conditioned" else None
    results = evaluate_model(model_dir, args.dataset, args.model_type, tokenizer_type, device)
    
    if results is None:
        print(f"Failed to evaluate model in {model_dir}")
        return
    
    evaluation_time = time.time() - start_time
    print(f"⏱️  Evaluation completed in {evaluation_time:.1f} seconds")
    
    print_evaluation_results(results, model_name)
    
    save_sample_images(
        results['generated_samples'], 
        results.get('sample_prompts'), 
        args.output_dir, 
        model_name
    )
    
    if 'training_stats' in results:
        create_training_curves_plot(results['training_stats'], args.output_dir, model_name)
    
    if args.save_json:
        json_results = {k: v for k, v in results.items() if k != 'generated_samples'}
        json_path = os.path.join(args.output_dir, f"{model_name}_evaluation.json")
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"JSON results saved to: {json_path}")
    
    print(f"\nAll results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()