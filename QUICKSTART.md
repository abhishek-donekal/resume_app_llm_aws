# Quick Start Guide

Get up and running with LLaMA 2 Resume Customizer in 5 minutes!

## 1️⃣ Setup (2 minutes)

```bash
# Setup environment
bash setup.sh

# Update .env with your settings
cp .env.example .env
nano .env  # Add your AWS credentials if using SageMaker
```

### Pick what the API will serve (MODEL_NAME)
In `.env`, set `MODEL_NAME` to one of:
- **Serve your fine-tuned local model** (recommended after training): `MODEL_NAME=./model_output`
- **Serve a HuggingFace model** (quick demo): `MODEL_NAME=meta-llama/Llama-2-7b-hf` (or `your-username/your-model`)

## 2️⃣ Prepare Training Data (1 minute)

```bash
# Sample data already included!
# To use your own data:
# 1. Create JSONL file in data/raw/your_data.jsonl
# 2. Each line: {"job_description": "...", "required_skills": [...], "current_resume": "...", "customized_resume": "..."}

# Prepare data
python data_prepare.py
```

## 3️⃣ Fine-tune Model (varies)

### Local/EC2 (Recommended for Testing)
```bash
# Start training (takes 1-2 hours on GPU)
python train_local.py --train-data data/processed/training_data.jsonl --output-dir ./model_output

# Or with bash script
bash train.sh --epochs 3 --batch-size 4 --output-dir ./model_output
```

### SageMaker (Production)
```bash
# Update .env with SageMaker details
# Then run:
python sagemaker_train.py
```

## 4️⃣ Run Inference

### Option A: FastAPI Server
```bash
# Start API server
bash run_api.sh

# Visit http://localhost:8000/docs for interactive API docs
```

**Test the API:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior Python Developer",
    "current_resume": "Base resume: 5 years backend experience building APIs and services...",
    "required_skills": ["Python", "FastAPI", "AWS"]
  }'
```

### Option B: Local Python Script
```bash
# Use fine-tuned model directly
python example_inference.py
```

## 5️⃣ Push to Git

```bash
git init
git add .
git commit -m "Initial LLaMA 2 resume customizer project"
git branch -M main
git remote add origin https://github.com/your-username/llama2-resume-customizer.git
git push -u origin main
```

---

## 📊 Project Structure

```
llama2-resume-customizer/
├── data/                    # Training data
├── model_output/            # Fine-tuned model (after training)
├── src/                     # Source code
├── scripts/                 # Utility scripts
├── aws/                     # AWS setup guides
├── README.md                # Full documentation
├── QUICKSTART.md            # This file
├── setup.sh                 # Initial setup
├── train.sh                 # Training script
├── run_api.sh               # API server
└── (no Docker required)
```

---

## 🚀 Common Commands

```bash
# Setup
bash setup.sh

# Prepare data
python data_prepare.py

# Train locally
bash train.sh

# Start API
bash run_api.sh

# Run inference
python example_inference.py

# Test API
python test_api.py
```

---

## 💡 Tips

1. **Start with sample data** to understand the flow
2. **Test locally first** before deploying to AWS
3. **Monitor training** with `tail -f logs/training.log`
4. **Use LoRA** for efficient fine-tuning (enabled by default)
5. **Check logs** if something goes wrong

---

## 🆘 Troubleshooting

### "Out of Memory" Error
```bash
# Reduce batch size in .env
BATCH_SIZE=2
```

### "Model not found" Error
```bash
# Download model first
huggingface-cli download meta-llama/Llama-2-7b-hf --cache-dir ./model_cache

# Requires HuggingFace login:
huggingface-cli login
```

### "API not responding"
```bash
# Check if server is running
curl http://localhost:8000/health
```

---

## 📚 Next Steps

1. Read [README.md](./README.md) for detailed documentation
2. Check [AWS_SETUP.md](./AWS_SETUP.md) for cloud deployment
3. Explore notebooks in `notebooks/` for detailed examples
4. Customize prompt template in `train_local.py`
5. Add your own training data

---

## 📞 Support

- 📖 Full docs: [README.md](./README.md)
- ☁️ AWS guide: [AWS_SETUP.md](./AWS_SETUP.md)
- 📝 Examples: [example_inference.py](./example_inference.py)
- 🧪 Tests: [test_api.py](./test_api.py)

---

**Happy training! 🚀**
