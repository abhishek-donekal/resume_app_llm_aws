# LLaMA 2 Resume Customizer

A production-grade project demonstrating **LLM fine-tuning and deployment** on AWS, featuring a custom resume generator that tailors resumes to specific job descriptions and required tools.

## 🎯 Project Overview

This project showcases:
- **LLM Fine-tuning**: Train LLaMA 2 on custom resume data
- **AWS Expertise**: Both managed (SageMaker) and self-managed (EC2) approaches
- **Multiple Deployments**: HuggingFace Hub, Ollama, FastAPI REST API
- **Production-Ready**: Docker, monitoring, error handling, logging
- **Scalability**: Ready for real-world use

### Key Features
✅ Fine-tune LLaMA 2 on resume/job description pairs  
✅ Generate customized resumes based on job requirements  
✅ Deploy on AWS with two different approaches  
✅ REST API for integration  
✅ Local deployment with Ollama  
✅ Model hosting on HuggingFace Hub  
✅ Complete documentation and examples  

---

## 📁 Project Structure

```
llama2-resume-customizer/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── Dockerfile                         # Docker containerization
│
├── data/
│   ├── raw/                          # Raw training data
│   │   └── sample_training_data.jsonl # Sample resume-job pairs
│   ├── processed/                    # Processed training data (after preparation)
│   └── prepare_data.py               # Data preparation script
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   ├── utils.py                      # Utility functions
│   ├── logger.py                     # Logging setup
│   │
│   ├── finetune/
│   │   ├── __init__.py
│   │   ├── sagemaker_train.py        # SageMaker fine-tuning
│   │   ├── ec2_train.py              # EC2-based fine-tuning
│   │   └── training_config.json      # Training hyperparameters
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── base_inference.py         # Base inference class
│   │   ├── local_inference.py        # Local model inference
│   │   └── huggingface_inference.py  # HuggingFace Hub inference
│   │
│   └── deployment/
│       ├── __init__.py
│       ├── fastapi_server.py         # FastAPI REST API
│       ├── ollama_setup.py           # Ollama deployment guide
│       └── docker_deployment.py      # Docker deployment utilities
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Explore training data
│   ├── 02_local_testing.ipynb        # Test fine-tuning locally
│   └── 03_inference_examples.ipynb   # Example inferences
│
├── aws/
│   ├── sagemaker_setup.sh            # SageMaker setup script
│   ├── ec2_setup.sh                  # EC2 setup script
│   ├── iam_policy.json               # IAM policy for AWS
│   └── README_AWS.md                 # Detailed AWS guide
│
├── tests/
│   ├── test_inference.py             # Test inference
│   ├── test_data_preparation.py      # Test data prep
│   └── test_api.py                   # Test FastAPI endpoints
│
└── scripts/
    ├── download_model.sh             # Download base LLaMA 2
    ├── train_local.sh                # Local training script
    ├── run_api.sh                    # Start FastAPI server
    └── push_to_huggingface.sh        # Push model to HuggingFace
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 8GB+ GPU memory (for local fine-tuning)
- AWS Account (for SageMaker/EC2 deployment)
- HuggingFace account (optional, for model hosting)

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/llama2-resume-customizer.git
cd llama2-resume-customizer
pip install -r requirements.txt
```

### 2. Prepare Training Data

```bash
python data/prepare_data.py --input data/raw/sample_training_data.jsonl --output data/processed/
```

### 3. Fine-tune Locally (Demo)

```bash
bash scripts/train_local.sh
```

### 4. Run Inference

```bash
python -m src.inference.local_inference \
  --model_path ./model_output \
  --job_description "Senior Python Developer required" \
  --skills "Python, FastAPI, AWS"
```

### 5. Deploy via FastAPI

```bash
bash scripts/run_api.sh
# Visit http://localhost:8000/docs for API documentation
```

---

## 🏗️ AWS Deployment Options

### Option 1: SageMaker (Managed)
**Pros**: Fully managed, automatic scaling, monitoring  
**Cons**: Higher cost

```bash
cd aws
bash sagemaker_setup.sh
python ../src/finetune/sagemaker_train.py
```

See [AWS Guide - SageMaker Section](./aws/README_AWS.md#sagemaker) for details.

### Option 2: EC2 (Self-Managed)
**Pros**: Cost-effective, full control  
**Cons**: Manual management

```bash
cd aws
bash ec2_setup.sh
ssh ec2-user@your-instance
python ../src/finetune/ec2_train.py
```

See [AWS Guide - EC2 Section](./aws/README_AWS.md#ec2) for details.

---

## 📦 Deployment Options

### Option 1: HuggingFace Hub (Easiest)
Host your fine-tuned model publicly or privately on HuggingFace.

```bash
bash scripts/push_to_huggingface.sh \
  --model_path ./model_output \
  --repo_name "your-username/llama2-resume-customizer" \
  --hf_token $HF_TOKEN
```

Then use it anywhere:
```python
from src.inference.huggingface_inference import HFInference
inference = HFInference("your-username/llama2-resume-customizer")
resume = inference.generate_resume(job_desc, skills)
```

### Option 2: Ollama (Local Deployment)
Perfect for testing and local use.

```bash
python src/deployment/ollama_setup.py
# Then query via Ollama
ollama pull your-model-name
ollama run your-model-name
```

### Option 3: FastAPI (REST API)
Production-ready REST API with Docker support.

```bash
bash scripts/run_api.sh
```

**Example Usage:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior Python Engineer",
    "current_skills": ["Python", "FastAPI"]
  }'
```

### Option 4: Docker Containerization
```bash
docker build -t llama2-resume-customizer .
docker run -p 8000:8000 llama2-resume-customizer
```

---

## 📊 Training Data Format

Training data should be JSONL format (one JSON per line):

```json
{
  "job_description": "Seeking Python Developer with AWS expertise",
  "required_skills": ["Python", "AWS", "FastAPI"],
  "current_resume": "John Doe. 5 years Python development...",
  "customized_resume": "John Doe. 5 years Python development with AWS expertise..."
}
```

See `data/raw/sample_training_data.jsonl` for examples.

---

## 🔧 Configuration

Copy `.env.example` to `.env` and update:

```env
# AWS
AWS_REGION=us-east-1
AWS_ROLE_ARN=arn:aws:iam::123456789:role/sagemaker-role

# HuggingFace
HF_TOKEN=your_huggingface_token

# Model
MODEL_NAME=meta-llama/Llama-2-7b-hf
BATCH_SIZE=4
EPOCHS=3
LEARNING_RATE=2e-5
```

---

## 📈 Model Performance

### Metrics Tracked
- Training loss
- Validation loss
- F1 score
- BLEU score (for text generation quality)
- Inference latency

### Sample Results
```
Epoch 1: Loss=2.45, Val_Loss=2.38
Epoch 2: Loss=1.89, Val_Loss=1.92
Epoch 3: Loss=1.45, Val_Loss=1.51

Inference Speed: ~2.3 tokens/sec (single GPU)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_inference.py

# With coverage
pytest --cov=src tests/
```

---

## 🎓 Learning Resources

- [Fine-tuning Guide](./docs/FINETUNING.md)
- [AWS Setup Guide](./aws/README_AWS.md)
- [API Documentation](./docs/API.md)
- [Deployment Guide](./docs/DEPLOYMENT.md)

---

## 📝 Example Notebooks

1. **01_data_exploration.ipynb** - Understand your data
2. **02_local_testing.ipynb** - Test fine-tuning locally before AWS
3. **03_inference_examples.ipynb** - See inference in action

---

## 🔐 Security & Best Practices

✅ Use `.env` for secrets (never commit!)  
✅ IAM roles for AWS access  
✅ Model versioning with git-lfs  
✅ Input validation on API endpoints  
✅ Rate limiting on production API  
✅ Monitoring and logging  

---

## 🚀 Next Steps

1. **Prepare Your Data**: Replace sample data with real training data
2. **Test Locally**: Run fine-tuning on your machine first
3. **Deploy to AWS**: Choose SageMaker or EC2 based on needs
4. **Monitor**: Track model performance in production
5. **Iterate**: Collect feedback and retrain

---

## 📚 Project Highlights for Interviews

🎯 **Technical Skills Demonstrated**:
- Large Language Models (LLaMA 2)
- Fine-tuning & Transfer Learning
- AWS cloud services (SageMaker, EC2)
- Python, FastAPI, Docker
- MLOps & Model Deployment
- REST APIs & Web Services

🎯 **Business Value**:
- Automated resume customization
- Reduced manual work
- Scalable solution
- Cost-effective deployment options

---

## 💼 Production Checklist

- [ ] Data quality validation
- [ ] Model evaluation metrics established
- [ ] Error handling & logging implemented
- [ ] API rate limiting configured
- [ ] Docker image built and tested
- [ ] AWS IAM policies secured
- [ ] Monitoring & alerts set up
- [ ] Documentation complete
- [ ] Unit tests passing
- [ ] Load testing completed

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 📧 Contact

Questions? Open an issue or reach out!

---

**Built with ❤️ to showcase practical LLM engineering**
