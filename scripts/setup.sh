#!/bin/bash

# Setup script for CLASS_robomimic
set -e  # Exit on any error

echo "Creating conda environment CLASS_robomimic with Python 3.11..."
conda create -n CLASS_robomimic python=3.11 -y

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate CLASS_robomimic

echo "Installing PyTorch and torchvision..."
pip install torch==2.6.0 torchvision==0.21.0

echo "Installing package in development mode..."
pip install -e .

echo "Setup complete! To use the environment, run:"
echo "conda activate CLASS_robomimic"