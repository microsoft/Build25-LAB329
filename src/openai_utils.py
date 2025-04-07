"""
Utility functions for interacting with the Azure OpenAI API.
"""
import os
import openai
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    TEMPERATURE
)

def initialize_openai_client():
    """Initialize the Azure OpenAI client."""
    openai.api_type = "azure"
    openai.api_key = AZURE_OPENAI_API_KEY
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_version = AZURE_OPENAI_API_VERSION
    
    client = openai.AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    
    return client

def generate_response(client, prompt, max_tokens=100):
    """
    Generate a response from the GPT-4o model.
    
    Args:
        client: Azure OpenAI client
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text response
    """
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=TEMPERATURE
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def generate_training_examples(client, prompts, max_tokens=100):
    """
    Generate training examples using GPT-4o as the teacher model.
    
    Args:
        client: Azure OpenAI client
        prompts: List of input prompts
        max_tokens: Maximum number of tokens to generate per response
        
    Returns:
        List of (prompt, response) pairs for training
    """
    examples = []
    for prompt in prompts:
        response = generate_response(client, prompt, max_tokens)
        if response:
            examples.append((prompt, response))
    
    return examples