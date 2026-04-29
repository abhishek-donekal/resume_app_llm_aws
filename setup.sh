#!/bin/bash

# Setup script for LLaMA 2 Resume Customizer
# Run this script to set up the project environment

set -e  # Exit on error

echo "🚀 Setting up LLaMA 2 Resume Customizer..."

# Check Python version
echo "📦 Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create directories
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed model_output model_cache logs

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please update .env with your configuration"
else
    echo "✅ .env file already exists"
fi

# Create virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo "🔧 Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created. Activate with: source venv/bin/activate"
fi

# Install dependencies
echo "⬇️  Installing dependencies..."
pip install -r requirements.txt

# Prepare sample data
echo "🔄 Preparing sample training data..."
python data_prepare.py \
    --input-dir data/raw/ \
    --output-dir data/processed/

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review and update .env file with your AWS credentials"
echo "2. Prepare your training data in data/raw/"
echo "3. Run: python data_prepare.py"
echo "4. Run: python train_local.py (for local training)"
echo "5. Run: bash scripts/run_api.sh (to start API server)"
echo ""
echo "For more details, see README.md"
