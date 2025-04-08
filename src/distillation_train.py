"""
Training script for model distillation in Azure AI Foundry using GPT-4o as the teacher model.
This script trains a smaller student model to mimic the outputs of GPT-4o.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from azureml.core import Run

from data_utils import load_examples_from_jsonl, prepare_dataset_from_examples
from config.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MAX_SEQ_LENGTH,
    STUDENT_MODEL_NAME,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    TEACHER_MODEL_NAME
)

def parse_args():
    parser = argparse.ArgumentParser(description="Model Distillation Training Script")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to use for training"
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="distilgpt2",  # A good starting point for distillation
        help="The student model architecture to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save the model"
    )
    parser.add_argument(
        "--use_azure_openai",
        action="store_true",
        help="Flag to use Azure OpenAI services"
    )
    return parser.parse_args()

def train():
    # Parse arguments
    args = parse_args()
    
    # Get the Azure ML run context
    run = Run.get_context()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Log Azure OpenAI configuration if using it
    if args.use_azure_openai:
        print(f"Using Azure OpenAI with deployment: {AZURE_OPENAI_DEPLOYMENT_NAME}")
        print(f"Azure OpenAI endpoint: {AZURE_OPENAI_ENDPOINT}")
        print(f"Teacher model: {TEACHER_MODEL_NAME}")
        
        # Set environment variables for Azure OpenAI
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = AZURE_OPENAI_ENDPOINT
        os.environ["OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
        os.environ["OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION
    
    # Load the student model and tokenizer
    print(f"Loading student model: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    model = AutoModelForCausalLM.from_pretrained(args.student_model)
    model.to(device)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    # In Azure ML, mounted datasets are typically available at /tmp/datasets/<dataset-name>
    dataset_path = f"/tmp/datasets/{args.dataset}/data.jsonl"
    examples = load_examples_from_jsonl(dataset_path)
    
    # Prepare dataloader
    dataloader = prepare_dataset_from_examples(
        examples, tokenizer, MAX_SEQ_LENGTH, BATCH_SIZE
    )
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Calculate total training steps
    total_steps = len(dataloader) * NUM_EPOCHS
    
    # Create scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"Starting training for {NUM_EPOCHS} epochs")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Log the loss
            run.log("batch_loss", loss.item())
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Log epoch loss
        avg_epoch_loss = total_loss / len(dataloader)
        run.log("epoch_loss", avg_epoch_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    final_output_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_output_dir, exist_ok=True)
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Saved final model to {final_output_dir}")
    
    # Register the model with Azure ML
    run.upload_folder("model", final_output_dir)
    run.register_model(
        model_name=STUDENT_MODEL_NAME,
        model_path="model",
        description="Distilled language model from GPT-4o"
    )
    print(f"Registered model as {STUDENT_MODEL_NAME}")

if __name__ == "__main__":
    train()