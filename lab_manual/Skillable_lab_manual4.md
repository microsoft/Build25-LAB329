
# Step 4: Register to Azure ML (5 min)

**Notebook:** `04.AzureML_RegisterToAzureML.ipynb`

**Purpose:** Register your optimized model to Azure ML for version tracking, sharing, and future deployment.

#### Instructions

1. **Open the notebook** from the file explorer.

---

# Model Registration to Azure Machine Learning

This notebook (`04.AzureML_RegisterToAzureML.ipynb`) implements the fourth phase of our model distillation pipeline: registering the optimized model to Azure Machine Learning. This step ensures the model is properly cataloged and available for deployment or sharing with other users.

## Purpose

This notebook demonstrates the model registration process by:
1. Connecting to an Azure Machine Learning workspace
2. Creating a model entity with appropriate metadata
3. Uploading the optimized ONNX model to the Azure ML model registry
4. Making the model available for future inference or deployment

## Workflow Overview

1. **Environment Setup**: Installing necessary packages and importing required libraries
2. **Authentication**: Connecting to Azure ML using the Azure Identity library
3. **Model Definition**: Creating a model entity with appropriate metadata
4. **Model Registration**: Uploading and registering the model to the Azure ML registry

## Technical Components

### Environment Setup
- Installation of required Python packages:
  - `python-dotenv` for environment variable management
  - Azure ML SDK libraries for model registration
- Importing necessary modules and classes for Azure ML interaction

### Authentication
- Loading authentication details from environment variables
- Using `DefaultAzureCredential` for secure authentication
- Creating an `MLClient` instance to interact with the Azure ML workspace

### Model Definition
- Creating a `Model` entity with the following properties:
  - Path to the optimized model from notebook 02
  - Asset type specification as a custom model
  - Model name and description for identification in the registry
- Setting appropriate metadata to make the model discoverable

### Model Registration
- Using the ML client to create or update the model in the registry
- Uploading the model files to Azure storage
- Registering the model with proper version management

## Code Highlights

```python
# Azure ML setup
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import os
from dotenv import load_dotenv

# Authentication
subscription_id = os.getenv('AZUREML_SUBSCRIPTION_ID')
resource_group = os.getenv('AZUREML_RESOURCE_GROUP')
workspace = os.getenv('AZUREML_WS_NAME')

ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

# Model definition
file_model = Model(
    path="models/phi-4-mini/onnx",
    type=AssetTypes.CUSTOM_MODEL,
    name="fine-tuning-phi-4-mini-onnx-int4-cpu",
    description="Fine tuning by MSOlive",
)

# Registration
ml_client.models.create_or_update(file_model)
```

## Benefits of This Approach

1. **Centralized Management**: Azure ML provides a central repository for all models
2. **Version Control**: Automatic versioning helps track model evolution
3. **Governance**: Maintains proper documentation and metadata for models
4. **Discoverability**: Makes models available to authorized team members
5. **Deployment Integration**: Simplifies the process of deploying the model to endpoints

This notebook completes the production preparation cycle by registering the optimized model in a central repository. By registering the model in Azure ML, we make it available for further deployment scenarios, sharing with team members, or integration into production pipelines. The registration process also ensures proper versioning and documentation, which is crucial for model governance and compliance.
