"""Setup script for cross-habitat marine species recognition."""

from setuptools import find_packages, setup

setup(
    name="cross-habitat-marine",
    version="1.0.0",
    description="Cross-habitat marine species recognition using vision foundation models",
    author="",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.56.0",
        "peft>=0.14.0",
        "timm>=1.0.20",
        "open-clip-torch>=2.26.0",
        "albumentations>=1.4.0",
        "scikit-learn>=1.4.0",
        "scipy>=1.12.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.66.0",
        "Pillow>=10.2.0",
        "opencv-python>=4.9.0",
    ],
    extras_require={
        "analysis": [
            "umap-learn>=0.5.0",
            "pycocotools>=2.0.7",
        ],
        "logging": [
            "wandb>=0.16.0",
            "codecarbon>=2.3.0",
        ],
    },
)
