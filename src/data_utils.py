"""
Utility functions for data processing and preparation for model distillation.
"""

import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class DistillationDataset(Dataset):
    """
    Dataset class for model distillation that pairs prompts with teacher model responses.
    """
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        prompt, response = self.examples[idx]
        
        # Tokenize the prompt (input to student model)
        prompt_encoding = self.tokenizer(
            prompt, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Tokenize the response (target output from teacher model)
        response_encoding = self.tokenizer(
            response,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels by shifting the response tokens
        labels = response_encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss calculation
        
        return {
            "input_ids": prompt_encoding["input_ids"].squeeze(),
            "attention_mask": prompt_encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

def prepare_dataset_from_examples(examples, tokenizer, max_length=512, batch_size=8):
    """
    Prepare a dataset and dataloader from examples.
    
    Args:
        examples: List of (prompt, response) pairs
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        batch_size: Batch size for training
        
    Returns:
        DataLoader for the dataset
    """
    dataset = DistillationDataset(examples, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def save_examples_to_jsonl(examples, output_path):
    """
    Save examples to a JSONL file.
    
    Args:
        examples: List of (prompt, response) pairs
        output_path: Path to save the JSONL file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt, response in examples:
            example = {
                "prompt": prompt,
                "response": response
            }
            f.write(json.dumps(example) + '\n')

def load_examples_from_jsonl(input_path):
    """
    Load examples from a JSONL file.
    
    Args:
        input_path: Path to the JSONL file
        
    Returns:
        List of (prompt, response) pairs
    """
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            examples.append((example["prompt"], example["response"]))
    
    return examples

def create_sample_prompts(num_samples=100):
    """
    Create sample prompts for generating training data.
    
    Args:
        num_samples: Number of sample prompts to generate
        
    Returns:
        List of sample prompts
    """
    # These are just examples - in practice, you'd want a diverse set of prompts
    # that cover the domain you're targeting with the distilled model
    sample_prompts = [
        "Explain the concept of machine learning to a 10-year old.",
        "Write a short poem about technology.",
        "What are the ethical implications of artificial intelligence?",
        "How does cloud computing work?",
        "Describe the water cycle in nature.",
        "What is the difference between supervised and unsupervised learning?",
        "Give me a recipe for chocolate chip cookies.",
        "Explain how to solve a Rubik's cube step by step.",
        "What are the main causes of climate change?",
        "How does blockchain technology work?"
    ]
    
    # Repeat or generate more prompts to reach num_samples
    if len(sample_prompts) < num_samples:
        # Simple approach: repeat the prompts
        sample_prompts = (sample_prompts * (num_samples // len(sample_prompts) + 1))[:num_samples]
    
    return sample_prompts