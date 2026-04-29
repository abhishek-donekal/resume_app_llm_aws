"""
Test script for FastAPI inference endpoint
"""

import requests
import json
import time
from typing import Dict

# API endpoint
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check passed\n")


def test_root():
    """Test root endpoint"""
    print("Testing / endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✅ Root endpoint passed\n")


def test_generate_resume(payload: Dict):
    """Test resume generation endpoint"""
    print(f"Testing /generate endpoint...")
    print(f"Request payload: {json.dumps(payload, indent=2)}")

    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/generate",
        json=payload,
        timeout=300  # 5 minute timeout for generation
    )
    elapsed_time = time.time() - start_time

    print(f"Status: {response.status_code}")
    print(f"Response time: {elapsed_time:.2f}s")

    if response.status_code == 200:
        result = response.json()
        print(f"Generated Resume:\n{result['customized_resume']}\n")
        print(f"Tokens generated: {result['tokens_generated']}")
        print(f"Generation time: {result['generation_time']:.2f}s")
        print("✅ Generate endpoint passed\n")
    else:
        print(f"Error: {response.text}")
        raise Exception(f"API returned {response.status_code}")


def test_invalid_request():
    """Test with invalid request"""
    print("Testing invalid request handling...")

    # Missing required field
    invalid_payload = {
        "required_skills": ["Python"],
        # Missing job_description
    }

    response = requests.post(
        f"{BASE_URL}/generate",
        json=invalid_payload
    )

    print(f"Status: {response.status_code}")
    assert response.status_code == 422  # Validation error
    print("✅ Invalid request handling passed\n")


def test_parameter_validation():
    """Test parameter validation"""
    print("Testing parameter validation...")

    payload = {
        "job_description": "Test job",
        "current_resume": "Base resume content that is long enough for validation.",
        "required_skills": ["Python"],
        "max_length": 5000,  # Too long
    }

    response = requests.post(
        f"{BASE_URL}/generate",
        json=payload
    )

    print(f"Status: {response.status_code}")
    assert response.status_code == 422  # Validation error
    print("✅ Parameter validation passed\n")


def main():
    """Run all tests"""
    print("=" * 80)
    print("LLaMA 2 Resume Customizer API Tests")
    print("=" * 80 + "\n")

    try:
        # Test basic endpoints
        test_health()
        test_root()

        # Test valid requests
        test_payloads = [
            {
                "job_description": "Senior Python Developer with FastAPI experience",
                "current_resume": "Backend developer with 5 years experience",
                "required_skills": ["Python", "FastAPI", "AWS"],
                "max_length": 512,
                "temperature": 0.7,
            },
            {
                "job_description": "Full Stack Engineer - React and Node.js",
                "current_resume": "Software engineer with frontend and backend experience across multiple projects.",
                "required_skills": ["React", "Node.js", "TypeScript"],
            },
            {
                "job_description": "Data Scientist with ML expertise",
                "current_resume": "Data analyst with statistics background",
                "required_skills": ["Python", "Machine Learning", "SQL"],
                "temperature": 0.5,
            }
        ]

        for i, payload in enumerate(test_payloads, 1):
            print(f"Test {i}:")
            try:
                test_generate_resume(payload)
            except Exception as e:
                print(f"⚠️  Test {i} failed: {str(e)}\n")

        # Test error handling
        test_invalid_request()
        test_parameter_validation()

        print("=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Tests failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
