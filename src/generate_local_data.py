"""
Generate training data locally for model distillation.
This script generates training examples using OpenAI models without requiring Azure OpenAI.
"""

import os
import argparse
import sys
import logging
from tqdm import tqdm
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import openai
from data_utils import create_sample_prompts, save_examples_to_jsonl

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Training Data for Model Distillation")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of training examples to generate"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/distillation_data.jsonl",
        help="Path to save the generated training data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use as the teacher"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate per response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for model responses"
    )
    parser.add_argument(
        "--api_key_env_var",
        type=str,
        default="OPENAI_API_KEY",
        help="Name of the environment variable containing the OpenAI API key"
    )
    return parser.parse_args()

def initialize_openai_client(api_key_env_var="OPENAI_API_KEY"):
    """Initialize the OpenAI client."""
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise ValueError(f"API key environment variable '{api_key_env_var}' not found")
    
    client = openai.OpenAI(api_key=api_key)
    return client

def generate_response(client, prompt, model="gpt-4o", max_tokens=100, temperature=0.7):
    """
    Generate a response from the OpenAI model.
    
    Args:
        client: OpenAI client
        prompt: Input text prompt
        model: Model name to use
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text response
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return None

def generate_training_examples(client, prompts, model="gpt-4o", max_tokens=100, temperature=0.7):
    """
    Generate training examples using the specified model as the teacher.
    
    Args:
        client: OpenAI client
        prompts: List of input prompts
        model: Model name to use
        max_tokens: Maximum number of tokens to generate per response
        temperature: Sampling temperature
        
    Returns:
        List of (prompt, response) pairs for training
    """
    examples = []
    
    for prompt in tqdm(prompts, desc="Generating responses"):
        response = generate_response(client, prompt, model, max_tokens, temperature)
        if response:
            examples.append((prompt, response))
    
    return examples

def main():
    # Load environment variables from .env file if present
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    logger.info(f"Initializing OpenAI client using {args.api_key_env_var} environment variable")
    
    try:
        # Initialize OpenAI client
        client = initialize_openai_client(args.api_key_env_var)
        
        # Create sample prompts
        logger.info(f"Creating {args.num_examples} sample prompts")
        prompts = create_sample_prompts(args.num_examples)
        
        # Generate responses
        logger.info(f"Generating responses using {args.model}")
        examples = generate_training_examples(
            client, 
            prompts,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        if not examples:
            logger.error("Failed to generate any training examples")
            return 1
        
        logger.info(f"Successfully generated {len(examples)} training examples")
        
        # Save examples to file
        output_dir = os.path.dirname(os.path.abspath(args.output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        save_examples_to_jsonl(examples, args.output_path)
        logger.info(f"Saved training examples to {args.output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())