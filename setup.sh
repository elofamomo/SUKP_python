#!/bin/bash
set -e

echo "=== Setting up SUKP training environment ==="

# Create required directories
mkdir -p checkpoints result figures metrics/logs

# Install PyTorch with CUDA (auto-detects CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install torch-geometric (matches installed PyTorch/CUDA)
pip install torch-geometric

# Install remaining dependencies
pip install numpy==1.26.0 matplotlib==3.10.9 PyYAML==6.0.3 tensorboard==2.20.0 mealpy==3.0.3

echo ""
echo "=== Setup complete. Verifying CUDA ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
md