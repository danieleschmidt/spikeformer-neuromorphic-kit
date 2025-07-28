#!/usr/bin/env python3
"""Setup script for SpikeFormer Neuromorphic Kit."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies
install_requires = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "pandas>=1.3.0",
    "tqdm>=4.62.0",
    "pyyaml>=6.0",
    "click>=8.0.0",
    "rich>=12.0.0",
    "transformers>=4.20.0",
    "datasets>=2.0.0",
    "accelerate>=0.20.0",
    "wandb>=0.15.0",
    "mlflow>=2.0.0",
    "prometheus-client>=0.14.0",
    "psutil>=5.8.0",
]

# Hardware-specific dependencies
loihi2_requires = [
    "nxsdk>=1.0.0",  # Intel Loihi SDK
]

spinnaker_requires = [
    "spynnaker>=6.0.0",
    "spalloc>=6.0.0",
]

edge_requires = [
    "onnx>=1.12.0",
    "onnxruntime>=1.12.0",
    "tensorrt>=8.4.0",
]

# Development dependencies
dev_requires = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "pytest-benchmark>=4.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=5.0.0",
    "mypy>=1.0.0",
    "ruff>=0.0.280",
    "pre-commit>=2.20.0",
    "bandit>=1.7.0",
    "safety>=2.0.0",
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "sphinx-autobuild>=2021.3.14",
    "jupyter>=1.0.0",
    "jupyterlab>=3.4.0",
    "notebook>=6.4.0",
    "ipywidgets>=7.7.0",
]

# Documentation dependencies
docs_requires = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "sphinx-autobuild>=2021.3.14",
    "myst-parser>=0.18.0",
    "sphinx-autodoc-typehints>=1.19.0",
]

# All extra dependencies
all_requires = loihi2_requires + spinnaker_requires + edge_requires + dev_requires

setup(
    name="spikeformer-neuromorphic-kit",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@your-org.com",
    description="Complete toolkit for training and deploying spiking transformer networks on neuromorphic hardware",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/spikeformer-neuromorphic-kit",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/spikeformer-neuromorphic-kit/issues",
        "Documentation": "https://docs.your-org.com/spikeformer",
        "Source Code": "https://github.com/your-org/spikeformer-neuromorphic-kit",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "loihi2": loihi2_requires,
        "spinnaker": spinnaker_requires,
        "edge": edge_requires,
        "dev": dev_requires,
        "docs": docs_requires,
        "all": all_requires,
    },
    entry_points={
        "console_scripts": [
            "spikeformer=spikeformer.cli:main",
            "spikeformer-convert=spikeformer.cli.convert:main",
            "spikeformer-train=spikeformer.cli.train:main",
            "spikeformer-deploy=spikeformer.cli.deploy:main",
            "spikeformer-profile=spikeformer.cli.profile:main",
        ],
    },
    include_package_data=True,
    package_data={
        "spikeformer": [
            "configs/*.yaml",
            "hardware/configs/*.json",
            "models/configs/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "neuromorphic",
        "spiking neural networks",
        "transformer",
        "energy efficient",
        "pytorch",
        "loihi",
        "spinnaker",
        "edge ai",
        "artificial intelligence",
        "machine learning",
    ],
)