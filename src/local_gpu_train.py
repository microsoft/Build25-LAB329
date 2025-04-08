"""
Local GPU training script for model distillation using A100 GPU.
This script trains a smaller Phi-4 student model to mimic the outputs of Llama-4-Scout-17B-16E without requiring Azure ML.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data_utils import load_examples_from_jsonl, DistillationDataset
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Local A100 GPU Model Distillation Training")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the JSONL file containing training data"
    )    parser.add_argument(
        "--student_model",
        type=str,
        default=STUDENT_MODEL_NAME,
        help="The student model architecture to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save the model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision training"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--use_azure_openai",
        action="store_true",
        help="Flag to use Azure OpenAI services for generating additional training data"
    )
    return parser.parse_args()

def setup_distributed(rank, world_size):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size, args):
    """Train the model using distributed data parallel on A100 GPU(s)."""
    # Set up distributed training if using multiple GPUs
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        logger.info(f"Using device: {device}")
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
            logger.info(f"CUDA Capability: {torch.cuda.get_device_capability(device)}")
    
    # Load the student model and tokenizer
    if rank == 0:
        logger.info(f"Loading student model: {args.student_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    model = AutoModelForCausalLM.from_pretrained(args.student_model)
    model.to(device)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    if rank == 0:
        logger.info(f"Loading dataset from: {args.dataset_path}")
    
    examples = load_examples_from_jsonl(args.dataset_path)
    if rank == 0:
        logger.info(f"Loaded {len(examples)} training examples")
    
    # Prepare dataset
    dataset = DistillationDataset(examples, tokenizer, max_length=args.max_seq_length)
    
    # Create sampler for distributed training
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    
    # Prepare dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None)
    )
    
    # Wrap model with DDP for multi-GPU training
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Calculate total training steps
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    
    # Create scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Set up mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Training loop
    if rank == 0:
        logger.info(f"Starting training for {args.num_epochs} epochs")
    
    for epoch in range(args.num_epochs):
        # Set sampler epoch for distributed training
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        
        # Create progress bar for the current epoch
        if rank == 0:
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        else:
            progress_bar = dataloader
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision if enabled
            if args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / args.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                # Standard forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / args.gradient_accumulation_steps
                
                # Standard backward pass
                loss.backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            # Track loss
            total_loss += loss.item() * args.gradient_accumulation_steps
            
            # Update progress bar on rank 0
            if rank == 0 and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})
        
        # Log epoch results on rank 0
        if rank == 0:
            avg_epoch_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint
            if world_size > 1:
                # For distributed training, save from rank 0 only and use the non-DDP model
                model_to_save = model.module
            else:
                model_to_save = model
                
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_to_save.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model on rank 0
    if rank == 0:
        if world_size > 1:
            model_to_save = model.module
        else:
            model_to_save = model
            
        final_output_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_output_dir, exist_ok=True)
        model_to_save.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        logger.info(f"Saved final model to {final_output_dir}")
    
    # Clean up distributed training
    if world_size > 1:
        dist.destroy_process_group()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure Azure OpenAI if requested
    if args.use_azure_openai:
        logger.info(f"Using Azure OpenAI with deployment: {AZURE_OPENAI_DEPLOYMENT_NAME}")
        logger.info(f"Azure OpenAI endpoint: {AZURE_OPENAI_ENDPOINT}")
        logger.info(f"Teacher model: {TEACHER_MODEL_NAME}")
        
        # Set environment variables for Azure OpenAI
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = AZURE_OPENAI_ENDPOINT
        os.environ["OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
        os.environ["OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION
    
    # Detect number of GPUs
    world_size = torch.cuda.device_count()
    logger.info(f"Detected {world_size} GPU(s)")
    
    if world_size > 1:
        # Use distributed training for multiple GPUs
        mp.spawn(
            train,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU or CPU training
        train(0, 1, args)

if __name__ == "__main__":
    main()