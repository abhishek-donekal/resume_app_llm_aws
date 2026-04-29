"""
Configuration management for LLaMA 2 Resume Customizer
Loads settings from environment variables with sensible defaults
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()


@dataclass
class AWSConfig:
    """AWS configuration"""
    region: str = os.getenv("AWS_REGION", "us-east-1")
    profile: str = os.getenv("AWS_PROFILE", "default")
    sagemaker_role_arn: str = os.getenv("SAGEMAKER_ROLE_ARN", "")
    sagemaker_instance_type: str = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.p3.2xlarge")
    sagemaker_instance_count: int = int(os.getenv("SAGEMAKER_INSTANCE_COUNT", "1"))
    sagemaker_bucket_name: str = os.getenv("SAGEMAKER_BUCKET_NAME", "sagemaker-bucket")


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
    model_id: str = os.getenv("MODEL_ID", "llama2-7b")
    output_dir: str = os.getenv("OUTPUT_DIR", "./model_output")
    cache_dir: str = os.getenv("CACHE_DIR", "./model_cache")
    max_seq_length: int = int(os.getenv("MAX_SEQ_LENGTH", "512"))


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = int(os.getenv("BATCH_SIZE", "4"))
    eval_batch_size: int = int(os.getenv("EVAL_BATCH_SIZE", "8"))
    epochs: int = int(os.getenv("EPOCHS", "3"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-5"))
    warmup_steps: int = int(os.getenv("WARMUP_STEPS", "100"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.01"))

    # LoRA (Parameter Efficient Fine-Tuning)
    use_lora: bool = True
    lora_r: int = int(os.getenv("LORA_R", "8"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "16"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))


@dataclass
class HuggingFaceConfig:
    """HuggingFace Hub configuration"""
    token: str = os.getenv("HF_TOKEN", "")
    repo_name: str = os.getenv("HF_REPO_NAME", "")
    private: bool = os.getenv("HF_PRIVATE", "false").lower() == "true"


@dataclass
class OllamaConfig:
    """Ollama configuration"""
    host: str = os.getenv("OLLAMA_HOST", "localhost:11434")
    model_name: str = os.getenv("OLLAMA_MODEL_NAME", "resume-customizer")


@dataclass
class APIConfig:
    """FastAPI configuration"""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    workers: int = int(os.getenv("API_WORKERS", "4"))
    log_level: str = os.getenv("LOG_LEVEL", "info")


@dataclass
class InferenceConfig:
    """Inference configuration"""
    max_length: int = int(os.getenv("INFERENCE_MAX_LENGTH", "512"))
    num_beams: int = int(os.getenv("INFERENCE_NUM_BEAMS", "4"))
    temperature: float = float(os.getenv("INFERENCE_TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("INFERENCE_TOP_P", "0.9"))


@dataclass
class DataConfig:
    """Data configuration"""
    input_path: str = os.getenv("DATA_INPUT_PATH", "./data/raw/")
    output_path: str = os.getenv("DATA_OUTPUT_PATH", "./data/processed/")
    test_size: float = float(os.getenv("TEST_SIZE", "0.1"))
    validation_size: float = float(os.getenv("VALIDATION_SIZE", "0.1"))


@dataclass
class Config:
    """Main configuration class"""
    # Sub-configs
    aws: AWSConfig = AWSConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    huggingface: HuggingFaceConfig = HuggingFaceConfig()
    ollama: OllamaConfig = OllamaConfig()
    api: APIConfig = APIConfig()
    inference: InferenceConfig = InferenceConfig()
    data: DataConfig = DataConfig()

    # General settings
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    seed: int = int(os.getenv("SEED", "42"))

    @property
    def device(self) -> str:
        """Get device (cuda or cpu)"""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


# Global config instance
config = Config()


if __name__ == "__main__":
    # Print configuration for debugging
    print("AWS Config:", config.aws)
    print("Model Config:", config.model)
    print("Training Config:", config.training)
    print("Device:", config.device)
