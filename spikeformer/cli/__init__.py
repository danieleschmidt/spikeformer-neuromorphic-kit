"""Command-line interface for SpikeFormer."""

from .main import main
from .convert import convert_main
from .train import train_main
from .deploy import deploy_main
from .profile import profile_main

__all__ = [
    "main",
    "convert_main", 
    "train_main",
    "deploy_main",
    "profile_main"
]