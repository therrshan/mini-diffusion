from setuptools import setup, find_packages

setup(
    name="mini-text-to-image-diffusion",
    version="0.1.0",
    description="A simplified implementation of a text-to-image diffusion model",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.21.0",
        "diffusers>=0.21.0",
        "datasets>=2.14.0",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "tensorboard>=2.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
        "evaluation": [
            "clip-by-openai>=1.0.0",
            "scipy>=1.9.0",
            "scikit-learn>=1.1.0",
        ],
    },
    python_requires=">=3.8",
)