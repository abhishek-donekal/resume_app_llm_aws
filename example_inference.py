"""
Example inference script for resume customization
Shows how to use the model for inference
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path


def load_model(model_path: str):
    """Load fine-tuned model"""
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.eval()
    print("✅ Model loaded successfully")
    return model, tokenizer


def generate_resume(
    model,
    tokenizer,
    job_description: str,
    current_resume: str,
    skills: list,
    max_length: int = 512,
):
    """Generate customized resume"""

    # Create prompt
    skills_str = ", ".join(skills)
    prompt = f"""You are a professional resume writer. Customize the following resume to match the job requirements:

Job Description: {job_description}
Required Skills: {skills_str}
Current Resume: {current_resume}

Customized Resume:"""

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    # Generate
    print("\n⏳ Generating resume...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            num_beams=4,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # Extract generated part
    customized_resume = generated_text.split("Customized Resume:")[-1].strip()

    return customized_resume


def main():
    # Model path (change this to your fine-tuned model path)
    model_path = "./model_output"

    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("Please fine-tune the model first using: python train_local.py")
        return

    # Load model
    model, tokenizer = load_model(model_path)

    # Example 1: Senior Python Developer
    print("\n" + "=" * 80)
    print("Example 1: Senior Python Developer Position")
    print("=" * 80)

    job_description_1 = """
    Senior Python Developer at TechCorp

    Requirements:
    - 5+ years Python development experience
    - FastAPI or Flask expertise
    - AWS cloud experience
    - Microservices architecture knowledge
    """

    current_resume_1 = """
    John Doe

    Experience:
    - Backend Developer at StartupX (2019-2023): Developed REST APIs using Flask
    - Junior Developer at WebCo (2018-2019): Built web applications with Django

    Skills: Python, JavaScript, HTML/CSS
    Education: BS Computer Science
    """

    skills_1 = ["Python", "FastAPI", "AWS", "Docker", "PostgreSQL"]

    resume_1 = generate_resume(
        model,
        tokenizer,
        job_description_1,
        current_resume_1,
        skills_1
    )

    print("\n📄 Generated Resume:")
    print("-" * 80)
    print(resume_1)
    print("-" * 80)

    # Example 2: Full Stack Engineer
    print("\n" + "=" * 80)
    print("Example 2: Full Stack Engineer Position")
    print("=" * 80)

    job_description_2 = """
    Full Stack Engineer at DataInnovate

    Looking for:
    - React and Node.js expertise
    - TypeScript proficiency
    - Cloud deployment experience
    - Machine learning pipeline knowledge
    """

    current_resume_2 = """
    Jane Smith

    Experience:
    - Frontend Developer at DesignStudio (2020-2023): Created user interfaces with React
    - Junior Frontend Dev at WebDesign Inc (2019-2020): Built responsive websites

    Skills: React, JavaScript, CSS, HTML
    Education: BA Information Technology
    """

    skills_2 = ["React", "Node.js", "TypeScript", "AWS", "MongoDB"]

    resume_2 = generate_resume(
        model,
        tokenizer,
        job_description_2,
        current_resume_2,
        skills_2
    )

    print("\n📄 Generated Resume:")
    print("-" * 80)
    print(resume_2)
    print("-" * 80)

    # Example 3: DevOps Engineer
    print("\n" + "=" * 80)
    print("Example 3: DevOps Engineer Position")
    print("=" * 80)

    job_description_3 = """
    DevOps Engineer at CloudSys

    Seeking:
    - Kubernetes and Docker expertise
    - Infrastructure as Code (Terraform)
    - CI/CD pipeline design
    - AWS or GCP experience
    """

    current_resume_3 = """
    Mike Johnson

    Experience:
    - Systems Administrator at TechServe (2018-2022): Managed servers and networks
    - IT Support at HelpDesk (2016-2018): Provided technical support

    Skills: Linux, Bash, Networking, Windows Server
    Education: Associate Degree in IT
    """

    skills_3 = ["Kubernetes", "Docker", "Terraform", "AWS", "CI/CD", "Linux"]

    resume_3 = generate_resume(
        model,
        tokenizer,
        job_description_3,
        current_resume_3,
        skills_3
    )

    print("\n📄 Generated Resume:")
    print("-" * 80)
    print(resume_3)
    print("-" * 80)

    print("\n✅ All examples completed!")


if __name__ == "__main__":
    main()
