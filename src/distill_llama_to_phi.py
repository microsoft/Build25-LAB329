"""
Main script for knowledge distillation from Llama-4-Scout-17B-16E to Phi-4 in Azure AI Foundry portal.

This script performs the following steps:
1. Generates training examples using Llama-4-Scout-17B-16E as the teacher model
2. Prepares and registers the dataset in Azure ML
3. Submits a distillation job to Azure AI Foundry using Phi-4 as the student model
4. Monitors the job and registers the final model

Usage:
    python distill_llama_to_phi.py [--num_examples NUM_EXAMPLES] [--student_model STUDENT_MODEL]
"""

import os
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai_utils import initialize_openai_client, generate_training_examples
from azure_ml_utils import get_ml_client, register_dataset, create_distillation_job
from data_utils import create_sample_prompts, save_examples_to_jsonl

from config.config import (
    DATASET_NAME,
    EXPERIMENT_NAME,
    TEACHER_MODEL_NAME,
    STUDENT_MODEL_NAME
)

def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation from Llama-4-Scout-17B-16E to Phi-4")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of training examples to generate"
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="Phi-4",  # Default student model architecture
        help="Student model architecture to use for distillation"
    )
    parser.add_argument(
        "--compute_target",
        type=str,
        default="gpu-cluster",  # Default compute target in Azure ML
        help="Compute target to use for training"
    )
    return parser.parse_args()

def main():
    # Load environment variables from .env file if present
    load_dotenv()
    
    # Parse arguments
    args = parse_args()
      print("Starting knowledge distillation process from Llama-4-Scout-17B-16E to Phi-4 in Azure AI Foundry...")
    
    # Step 1: Generate training examples using Llama-4-Scout-17B-16E
    print(f"Generating {args.num_examples} training examples using {TEACHER_MODEL_NAME}...")
    
    # Initialize Azure OpenAI client
    openai_client = initialize_openai_client()
    if not openai_client:
        print("Failed to initialize Azure OpenAI client. Please check your configuration.")
        return
    
    # Create sample prompts
    prompts = create_sample_prompts(args.num_examples)
    
    # Generate responses from the teacher model (Llama-4-Scout)
    examples = generate_training_examples(openai_client, prompts)
    if not examples:
        print("Failed to generate training examples. Please check your Azure OpenAI configuration.")
        return
    
    print(f"Generated {len(examples)} training examples.")
    
    # Save examples to file
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    dataset_path = os.path.join(data_dir, "distillation_data.jsonl")
    save_examples_to_jsonl(examples, dataset_path)
    print(f"Saved training examples to {dataset_path}")
    
    # Step 2: Register dataset in Azure ML
    print("Initializing Azure ML client...")
    ml_client = get_ml_client()
    if not ml_client:
        print("Failed to initialize Azure ML client. Please check your configuration.")
        return
    
    print(f"Registering dataset as '{DATASET_NAME}' in Azure ML...")
    dataset = register_dataset(ml_client, data_dir, DATASET_NAME)
    if not dataset:
        print("Failed to register dataset in Azure ML. Please check your Azure ML configuration.")
        return
    
    print(f"Successfully registered dataset: {dataset.name}, version: {dataset.version}")
    
    # Step 3: Submit distillation job to Azure AI Foundry
    print("Submitting distillation job to Azure AI Foundry...")
    
    # The source directory should include all necessary scripts for training
    source_dir = os.path.dirname(os.path.abspath(__file__))
    
    job = create_distillation_job(
        ml_client,
        DATASET_NAME,
        EXPERIMENT_NAME,
        source_dir,
        entry_script="distillation_train.py",
        compute_target=args.compute_target
    )
    
    if not job:
        print("Failed to create distillation job. Please check your Azure ML configuration.")
        return
    
    print(f"Distillation job created with ID: {job.id}")
    print(f"You can monitor the job in the Azure AI Foundry portal.")
    print(f"Once completed, the distilled model will be registered as '{STUDENT_MODEL_NAME}'.")

if __name__ == "__main__":
    main()
