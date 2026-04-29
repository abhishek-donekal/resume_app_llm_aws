"""
Fine-tune LLaMA 2 on SageMaker
Uses managed SageMaker training job
"""

import json
import logging
from pathlib import Path
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.utils import name_from_base

from config import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SageMakerTrainer:
    """Manages SageMaker training job for LLaMA 2 fine-tuning"""

    def __init__(self):
        self.sagemaker_session = sagemaker.Session()
        self.role = config.aws.sagemaker_role_arn
        self.bucket = config.aws.sagemaker_bucket_name
        self.region = config.aws.region

    def upload_data_to_s3(self, local_path: str, s3_prefix: str) -> str:
        """Upload training data to S3"""
        logger.info(f"Uploading data from {local_path} to S3...")
        s3_path = self.sagemaker_session.upload_data(
            path=local_path,
            bucket=self.bucket,
            key_prefix=s3_prefix
        )
        logger.info(f"Data uploaded to {s3_path}")
        return s3_path

    def create_training_script(self) -> str:
        """Create training script for SageMaker"""
        script_content = '''
import json
import os
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


class ResumeDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['prompt'] + item['completion']

        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--training_dir", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)

    args = parser.parse_args()

    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    logger.info("Loading training data...")
    train_data_path = Path(args.training_dir) / "training_data.jsonl"
    dataset = ResumeDataset(train_data_path, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_steps=100,
        logging_steps=50,
        logging_dir=f"{args.output_dir}/logs",
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
'''
        return script_content

    def start_training_job(self, training_data_s3_path: str, job_name: str = None):
        """Start SageMaker training job"""
        if not job_name:
            job_name = name_from_base("llama2-resume")

        logger.info(f"Starting SageMaker training job: {job_name}")

        # Write training script
        script_path = Path("train.py")
        script_path.write_text(self.create_training_script())

        # Create estimator
        estimator = Estimator(
            image_uri=self._get_training_image(),
            role=self.role,
            instance_count=config.aws.sagemaker_instance_count,
            instance_type=config.aws.sagemaker_instance_type,
            output_path=f"s3://{self.bucket}/output",
            code_location=f"s3://{self.bucket}/code",
            sagemaker_session=self.sagemaker_session,
        )

        # Set hyperparameters
        estimator.set_hyperparameters(
            model_name=config.model.model_name,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            lr=config.training.learning_rate,
        )

        # Start training
        estimator.fit(
            {"training": training_data_s3_path},
            job_name=job_name,
        )

        logger.info(f"Training job started: {job_name}")
        return job_name

    def _get_training_image(self) -> str:
        """Get appropriate training image for region"""
        # PyTorch training image
        region = self.region
        return (
            f"763104330519.dkr.ecr.{region}.amazonaws.com/"
            "pytorch-training:2.0.1-gpu-py310"
        )


if __name__ == "__main__":
    trainer = SageMakerTrainer()

    # Upload data
    s3_data_path = trainer.upload_data_to_s3(
        "data/processed/",
        "training-data"
    )

    # Start training
    job_name = trainer.start_training_job(s3_data_path)
    print(f"Training job: {job_name}")
