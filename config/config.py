"""
Configuration settings for the distillation project using Azure AI Foundry portal and OpenAI GPT-4o.
"""

# Azure AI Foundry Configuration
AZURE_ML_SUBSCRIPTION_ID = "your-subscription-id"
AZURE_ML_RESOURCE_GROUP = "your-resource-group"
AZURE_ML_WORKSPACE_NAME = "your-workspace-name"

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = "https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_OPENAI_API_VERSION = "2023-12-01-preview"  # Update with the latest API version
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"  # Your GPT-4o deployment name

# Distillation Configuration
TEACHER_MODEL_NAME = "gpt-4o"  # The teacher model (GPT-4o)
STUDENT_MODEL_NAME = "DistilledModel"  # Name for your distilled model
DATASET_NAME = "distillation_dataset"  # Name for your training dataset
EXPERIMENT_NAME = "gpt4o-distillation"  # Name for your experiment in Azure ML

# Training Parameters
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512
TEMPERATURE = 0.7