"""
Local/EC2 fine-tuning of LLaMA 2 for resume customization
Run this directly on EC2 or locally
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TextDataset,
)
from peft import LoraConfig, get_peft_model
import wandb

from config import config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ResumeDataset(Dataset):
    """Dataset for resume customization"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: List[Dict] = []

        # Load JSONL
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        text = item['prompt'] + item['completion']

        # Tokenize
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


class LLaMATrainer:
    """Fine-tunes LLaMA 2 model"""

    def __init__(self, config_obj=config):
        self.config = config_obj
        self.device = torch.device(config_obj.device)
        self.model = None
        self.tokenizer = None

        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load base model and tokenizer"""
        logger.info(f"Loading model: {self.config.model.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            cache_dir=self.config.model.cache_dir,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=self.config.model.cache_dir,
            trust_remote_code=True,
        )

        logger.info(f"Model loaded successfully")

    def apply_lora(self):
        """Apply LoRA for parameter-efficient fine-tuning"""
        if not self.config.training.use_lora:
            logger.info("LoRA disabled, using full fine-tuning")
            return

        logger.info("Applying LoRA configuration...")

        lora_config = LoraConfig(
            r=self.config.training.lora_r,
            lora_alpha=self.config.training.lora_alpha,
            lora_dropout=self.config.training.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train(self, train_data_path: str, eval_data_path: str = None):
        """Fine-tune the model"""

        # Load datasets
        logger.info("Loading training dataset...")
        train_dataset = ResumeDataset(
            train_data_path,
            self.tokenizer,
            self.config.model.max_seq_length
        )

        eval_dataset = None
        if eval_data_path and Path(eval_data_path).exists():
            logger.info("Loading evaluation dataset...")
            eval_dataset = ResumeDataset(
                eval_data_path,
                self.tokenizer,
                self.config.model.max_seq_length
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.model.output_dir,
            num_train_epochs=self.config.training.epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.eval_batch_size,
            learning_rate=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            weight_decay=self.config.training.weight_decay,
            save_strategy="steps",
            save_steps=100,
            logging_steps=50,
            logging_dir=f"{self.config.model.output_dir}/logs",
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            report_to=["wandb"] if self.config.debug else [],
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Train
        logger.info("Starting fine-tuning...")
        trainer.train()

        logger.info(f"Training complete! Model saved to {self.config.model.output_dir}")

    def save_model(self):
        """Save fine-tuned model"""
        logger.info(f"Saving model to {self.config.model.output_dir}")
        Path(self.config.model.output_dir).mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(self.config.model.output_dir)
        self.tokenizer.save_pretrained(self.config.model.output_dir)

        logger.info("Model saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 2 on resume data")
    parser.add_argument("--train-data", type=str, default="data/processed/training_data.jsonl",
                        help="Training data path")
    parser.add_argument("--eval-data", type=str, default=None,
                        help="Evaluation data path (optional)")
    parser.add_argument("--output-dir", type=str, default="./model_output",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--no-lora", action="store_true",
                        help="Disable LoRA")

    args = parser.parse_args()

    # Override config with CLI args
    if args.output_dir:
        config.model.output_dir = args.output_dir
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.no_lora:
        config.training.use_lora = False

    # Initialize trainer
    trainer = LLaMATrainer(config)

    try:
        # Load model
        trainer.load_model()

        # Apply LoRA
        trainer.apply_lora()

        # Train
        trainer.train(args.train_data, args.eval_data)

        # Save
        trainer.save_model()

        logger.info("✅ Fine-tuning complete!")

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
