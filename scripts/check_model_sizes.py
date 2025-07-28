#!/usr/bin/env python3
"""Check model file sizes to prevent committing large models."""

import os
import sys
from pathlib import Path


def check_model_sizes(max_size_mb: int = 100) -> bool:
    """Check for large model files that shouldn't be committed.
    
    Args:
        max_size_mb: Maximum allowed file size in MB
        
    Returns:
        True if all files are within size limits, False otherwise
    """
    repo_root = Path.cwd()
    large_files = []
    
    # Model file extensions to check
    model_extensions = ['.pth', '.pt', '.ckpt', '.h5', '.pb', '.onnx', '.pkl', '.pickle']
    
    for ext in model_extensions:
        for file_path in repo_root.rglob(f'*{ext}'):
            # Skip files in .git directory
            if '.git' in file_path.parts:
                continue
                
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > max_size_mb:
                large_files.append((file_path, file_size_mb))
    
    if large_files:
        print("❌ Large model files detected:")
        for file_path, size_mb in large_files:
            print(f"  {file_path}: {size_mb:.1f} MB")
        print(f"\nFiles larger than {max_size_mb} MB should not be committed.")
        print("Consider using Git LFS or exclude them in .gitignore")
        return False
    
    print("✅ No large model files detected")
    return True


if __name__ == "__main__":
    success = check_model_sizes()
    sys.exit(0 if success else 1)