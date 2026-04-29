#!/bin/bash

# Fine-tune LLaMA 2 locally or on EC2
# Usage: bash train.sh [--sagemaker] [--epochs 3] [--batch-size 4]

set -e

echo "🚀 Starting LLaMA 2 Fine-tuning..."

# Parse arguments
use_sagemaker=false
epochs=3
batch_size=4
learning_rate="2e-5"
output_dir="./model_output"

while [[ $# -gt 0 ]]; do
    case $1 in
        --sagemaker)
            use_sagemaker=true
            shift
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        --batch-size)
            batch_size="$2"
            shift 2
            ;;
        --lr|--learning-rate)
            learning_rate="$2"
            shift 2
            ;;
        --output-dir)
            output_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Prepare data
echo "📊 Preparing training data..."
python data_prepare.py \
    --input-dir data/raw/ \
    --output-dir data/processed/

if [ "$use_sagemaker" = true ]; then
    echo "☁️  Starting SageMaker training job..."
    python sagemaker_train.py
else
    echo "💻 Starting local fine-tuning..."
    python train_local.py \
        --train-data data/processed/training_data.jsonl \
        --output-dir "$output_dir" \
        --epochs "$epochs" \
        --batch-size "$batch_size" \
        --learning-rate "$learning_rate"
fi

echo ""
echo "✅ Fine-tuning complete!"
echo "Model saved to: $output_dir"
echo ""
echo "To start the API server, run:"
echo "  bash run_api.sh"
