"""
Data preparation script for resume customization training
Processes raw JSONL data into training format
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparator:
    """Prepares training data for fine-tuning"""

    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def create_prompt(self, job_description: str, required_skills: List[str]) -> str:
        """Create prompt for fine-tuning"""
        skills_str = ", ".join(required_skills)
        prompt = f"""You are a professional resume writer. Given a job description and required skills, customize a resume to highlight relevant experience.

Job Description: {job_description}
Required Skills: {skills_str}

Customize the resume:"""
        return prompt

    def create_training_samples(self, raw_data: List[Dict]) -> List[Dict]:
        """Convert raw data to training samples"""
        samples = []
        for item in raw_data:
            prompt = self.create_prompt(
                item['job_description'],
                item.get('required_skills', [])
            )
            sample = {
                "prompt": prompt,
                "completion": item['customized_resume'],
                "job_id": item.get('job_id', ''),
            }
            samples.append(sample)
        return samples

    def save_jsonl(self, data: List[Dict], output_file: Path):
        """Save data as JSONL"""
        with open(output_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved {len(data)} samples to {output_file}")

    def process(self, input_file: str):
        """Process raw data file"""
        logger.info(f"Loading data from {input_file}")
        raw_data = self.load_jsonl(self.input_path / input_file)
        logger.info(f"Loaded {len(raw_data)} raw samples")

        logger.info("Creating training samples...")
        samples = self.create_training_samples(raw_data)

        output_file = self.output_path / "training_data.jsonl"
        self.save_jsonl(samples, output_file)

        logger.info(f"Data preparation complete!")
        logger.info(f"Total samples: {len(samples)}")
        return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--input", type=str, default="data/raw/sample_training_data.jsonl",
                        help="Input JSONL file")
    parser.add_argument("--input-dir", type=str, default="data/raw/",
                        help="Input directory")
    parser.add_argument("--output-dir", type=str, default="data/processed/",
                        help="Output directory")

    args = parser.parse_args()

    preparator = DataPreparator(args.input_dir, args.output_dir)
    input_filename = Path(args.input).name
    preparator.process(input_filename)
