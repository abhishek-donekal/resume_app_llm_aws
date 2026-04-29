"""
FastAPI REST API for resume customization inference
Production-ready API with validation and error handling
"""

import logging
from typing import Optional, List
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import asynccontextmanager

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and tokenizer
model = None
tokenizer = None


class ResumeRequest(BaseModel):
    """Resume generation request"""
    job_description: str = Field(..., min_length=10, description="Job description")
    current_resume: Optional[str] = Field(None, description="Current resume content")
    required_skills: List[str] = Field(
        default_factory=list,
        description="Required skills from job"
    )
    max_length: Optional[int] = Field(
        default=512,
        ge=100,
        le=2048,
        description="Maximum length of generated text"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for generation"
    )


class ResumeResponse(BaseModel):
    """Resume generation response"""
    customized_resume: str
    job_description: str
    required_skills: List[str]
    generation_time: float
    tokens_generated: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str


def load_model_and_tokenizer():
    """Load model and tokenizer"""
    global model, tokenizer

    logger.info(f"Loading model: {config.model.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        cache_dir=config.model.cache_dir,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    device = torch.device(config.device)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=config.model.cache_dir,
        trust_remote_code=True,
    )

    model.eval()
    logger.info("Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    logger.info("Starting up...")
    load_model_and_tokenizer()
    logger.info("Application ready")
    yield
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="LLaMA 2 Resume Customizer API",
    description="Fine-tuned LLaMA 2 for resume customization",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": config.device,
    }


@app.post("/generate", response_model=ResumeResponse, tags=["Inference"])
async def generate_resume(request: ResumeRequest):
    """Generate customized resume"""

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please try again later."
        )

    try:
        import time
        start_time = time.time()

        # Create prompt
        skills_str = ", ".join(request.required_skills) if request.required_skills else "N/A"

        prompt = f"""You are a professional resume writer. Customize the following resume to match the job requirements:

Job Description: {request.job_description}
Required Skills: {skills_str}
Current Resume: {request.current_resume or 'Not provided'}

Customized Resume:"""

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.model.max_seq_length
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=config.inference.top_p,
                num_beams=config.inference.num_beams,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Extract only the generated part (after prompt)
        customized_resume = generated_text.split("Customized Resume:")[-1].strip()

        generation_time = time.time() - start_time
        tokens_generated = outputs[0].shape[0] - inputs['input_ids'].shape[1]

        logger.info(f"Generated resume in {generation_time:.2f}s")

        return {
            "customized_resume": customized_resume,
            "job_description": request.job_description,
            "required_skills": request.required_skills,
            "generation_time": generation_time,
            "tokens_generated": tokens_generated,
        }

    except Exception as e:
        logger.error(f"Generation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating resume: {str(e)}"
        )


@app.get("/", tags=["Root"])
async def root():
    """API documentation"""
    return {
        "message": "LLaMA 2 Resume Customizer API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "generate": {
                "path": "/generate",
                "method": "POST",
                "description": "Generate customized resume",
            }
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        log_level=config.api.log_level,
    )
