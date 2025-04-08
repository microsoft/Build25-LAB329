"""
Configuration settings for the distillation project using Microsoft Azure AI Foundry portal with Llama-4-Scout-17B-16E as teacher and Phi-4 as student.
"""

# Microsoft Azure AI Foundry Configuration
AZURE_ML_SUBSCRIPTION_ID = "your-subscription-id"
AZURE_ML_RESOURCE_GROUP = "your-resource-group"
AZURE_ML_WORKSPACE_NAME = "your-workspace-name"

# Microsoft Azure AI Foundry Model Configuration
AZURE_OPENAI_ENDPOINT = "https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_OPENAI_API_VERSION = "2023-12-01-preview"  # Update with the latest API version
AZURE_OPENAI_DEPLOYMENT_NAME = "llama-scout-teacher"  # Your Llama-4-Scout-17B-16E deployment name

# Distillation Configuration
TEACHER_MODEL_NAME = "Llama-4-Scout-17B-16E"  # The teacher model
STUDENT_MODEL_NAME = "Phi-4"  # The student model
DATASET_NAME = "distillation_dataset"  # Name for your training dataset
EXPERIMENT_NAME = "llama-to-phi-distillation"  # Name for your experiment in Azure ML

# Training Parameters
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512
TEMPERATURE = 0.7