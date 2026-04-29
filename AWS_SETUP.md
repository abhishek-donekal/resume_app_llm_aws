# AWS Deployment Guide for LLaMA 2 Resume Customizer

This guide covers deploying your fine-tuned model on AWS using both SageMaker and EC2.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [SageMaker Setup](#sagemaker-setup)
3. [EC2 Setup](#ec2-setup)
4. [Cost Estimation](#cost-estimation)
5. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Prerequisites

### AWS Account Setup
1. Create an AWS account at https://aws.amazon.com
2. Install AWS CLI:
   ```bash
   pip install awscli
   ```

3. Configure AWS credentials:
   ```bash
   aws configure
   # Enter your AWS Access Key ID and Secret Access Key
   ```

4. Create an S3 bucket for training data:
   ```bash
   aws s3 mb s3://your-bucket-name-llama2-resume
   ```

---

## SageMaker Setup

### Option A: Using SageMaker (Recommended for Production)

**Advantages:**
- ✅ Fully managed service
- ✅ Automatic scaling
- ✅ Built-in monitoring
- ✅ Easy to deploy endpoints
- ✅ Integration with other AWS services

### Step 1: Create SageMaker Execution Role

```bash
# Create IAM role
aws iam create-role \
  --role-name SageMakerLLaMA2Role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {
          "Service": "sagemaker.amazonaws.com"
        },
        "Action": "sts:AssumeRole"
      }
    ]
  }'

# Attach policies
aws iam attach-role-policy \
  --role-name SageMakerLLaMA2Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name SageMakerLLaMA2Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Step 2: Update Configuration

Edit `.env`:
```env
SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerLLaMA2Role
SAGEMAKER_BUCKET_NAME=your-bucket-name-llama2-resume
SAGEMAKER_INSTANCE_TYPE=ml.p3.2xlarge
SAGEMAKER_INSTANCE_COUNT=1
```

### Step 3: Upload Training Data

```bash
# Prepare data locally first
python data_prepare.py

# Upload to S3
aws s3 cp data/processed/ s3://your-bucket-name-llama2-resume/training-data/ --recursive
```

### Step 4: Start SageMaker Training Job

```bash
python sagemaker_train.py
```

Monitor your training job:
```bash
# Via CLI
aws sagemaker describe-training-job \
  --training-job-name llama2-resume-XXXXX

# Via AWS Console
# Open https://console.aws.amazon.com/sagemaker/
# Navigate to Training Jobs
```

### Step 5: Create SageMaker Endpoint (for Inference)

```python
import sagemaker
from sagemaker.model import Model

# Get output model location from training job
model_uri = "s3://your-bucket/path-to-model.tar.gz"

# Create model
model = Model(
    image_uri=f"763104330519.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
    model_data=model_uri,
    role=sagemaker_role_arn,
)

# Deploy endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.p3.2xlarge",
    endpoint_name="llama2-resume-endpoint"
)

# Invoke endpoint
response = predictor.predict({"job_description": "..."})
```

---

## EC2 Setup

### Option B: Using EC2 (Cost-Effective)

**Advantages:**
- ✅ Lower cost
- ✅ Full control
- ✅ Custom configuration
- ✅ Good for development/testing

### Step 1: Launch EC2 Instance

```bash
# Using AWS CLI
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}"
```

Or via AWS Console:
1. Go to EC2 Dashboard
2. Click "Launch Instance"
3. Select Ubuntu 22.04 AMI
4. Choose `g4dn.xlarge` instance type
5. Configure storage: 100 GB minimum
6. Add security group to allow SSH (port 22) and HTTP (port 8000)

### Step 2: Connect to Instance

```bash
ssh -i your-key.pem ec2-user@your-instance-ip
```

### Step 3: Install Dependencies

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install NVIDIA drivers
sudo apt-get install -y nvidia-driver-535

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Step 4: Clone Project & Setup

```bash
cd /home/ec2-user
git clone https://github.com/your-username/llama2-resume-customizer.git
cd llama2-resume-customizer

# Copy environment file
cp .env.example .env
nano .env  # Edit with your AWS credentials
```

### Step 5: Download Model & Prepare Data

```bash
# Download base LLaMA 2 model
huggingface-cli download meta-llama/Llama-2-7b-hf --cache-dir ./model_cache

# Prepare training data
python data_prepare.py
```

### Step 6: Start Training

```bash
# Start training in background
nohup python train_local.py \
  --train-data data/processed/training_data.jsonl \
  --output-dir ./model_output \
  > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### Step 7: Start API Server

```bash
# After training completes
bash run_api.sh &

# Or with nohup for persistence
nohup bash run_api.sh > api.log 2>&1 &
```

### Step 8: Connect Elastic IP (Optional)

```bash
# Allocate Elastic IP
aws ec2 allocate-address --domain vpc

# Associate with instance
aws ec2 associate-address \
  --instance-id i-xxxxxxxx \
  --public-ip xxx.xxx.xxx.xxx
```

---

## Docker Deployment (Both Platforms)

### Build Docker Image

```bash
docker build -t llama2-resume-customizer:latest .
```

### Run Docker Container

```bash
docker run \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/model_output:/app/model_output \
  -e HF_TOKEN=$HF_TOKEN \
  llama2-resume-customizer:latest
```

### Push to ECR (for SageMaker)

```bash
# Create ECR repository
aws ecr create-repository --repository-name llama2-resume-customizer

# Tag image
docker tag llama2-resume-customizer:latest \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/llama2-resume-customizer:latest

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/llama2-resume-customizer:latest
```

---

## Cost Estimation

### SageMaker Costs (per hour)
- Training: `ml.p3.2xlarge` = ~$3.06/hour
- Inference: `ml.p3.2xlarge` = ~$3.06/hour
- Storage: ~$0.023/GB/month

### EC2 Costs (per hour)
- `g4dn.xlarge`: ~$0.526/hour
- EBS Storage: ~$0.10/GB/month
- Data Transfer: varies

### Estimated Total (for 30-day development)

**SageMaker:**
- 100 hours training: $306
- 100 hours endpoint: $306
- Storage: $5
- **Total: ~$617**

**EC2:**
- 200 hours compute: $105
- Storage: $3
- Data transfer: $10
- **Total: ~$118**

---

## Monitoring & Maintenance

### CloudWatch Monitoring (SageMaker)

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Get training metrics
metrics = cloudwatch.get_metric_statistics(
    Namespace='AWS/SageMaker',
    MetricName='TrainingJobStatus',
    Dimensions=[
        {'Name': 'TrainingJobName', 'Value': 'your-job-name'}
    ],
    StartTime=datetime(2024, 1, 1),
    EndTime=datetime(2024, 1, 31),
    Period=3600,
    Statistics=['Average']
)
```

### EC2 Monitoring

```bash
# CPU usage
top

# GPU usage
nvidia-smi

# Disk space
df -h

# Memory usage
free -h
```

### Cost Optimization Tips

1. **Auto-Scaling**: Configure SageMaker auto-scaling
2. **Spot Instances**: Use EC2 Spot for 70% cost savings
3. **Reserved Instances**: Commit to 1-3 year terms
4. **Data Transfer**: Keep data in same region
5. **Stop Unused Resources**: Always stop endpoints when not in use

---

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
BATCH_SIZE=2  # from 4

# Use gradient accumulation
GRADIENT_ACCUMULATION_STEPS=2
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi -l 1

# Enable mixed precision
USE_AMP=true
```

### API Not Responding
```bash
# Check logs
journalctl -u uvicorn -n 100

# Restart API
systemctl restart uvicorn
```

---

## Production Checklist

- [ ] IAM roles and permissions configured
- [ ] S3 bucket created and secured
- [ ] Training data uploaded
- [ ] Cost monitoring enabled
- [ ] CloudWatch alarms configured
- [ ] Model versioning strategy
- [ ] Backup and recovery plan
- [ ] API rate limiting configured
- [ ] SSL/TLS certificates installed
- [ ] Auto-scaling configured

---

## Support & Resources

- **AWS SageMaker Docs**: https://docs.aws.amazon.com/sagemaker/
- **EC2 User Guide**: https://docs.aws.amazon.com/ec2/
- **AWS Pricing**: https://aws.amazon.com/pricing/
- **AWS Support**: https://console.aws.amazon.com/support/

---

**Last Updated**: April 2024
