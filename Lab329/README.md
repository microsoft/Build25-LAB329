# **Model Distillation and Azure AI Foundry Workshop**

## Workshop Overview

This is a practical workshop on model distillation using Azure AI Foundry. Through a series of notebooks, it demonstrates how to extract knowledge from LLMs and transfer it to SLMs while maintaining good performance. This process utilizes various features of the Azure Machine Learning (AzureML) platform, with a particular focus on optimizing models and deploying them to production environments.

## Folder Structure


- **Notebooks /**: Contains Jupyter notebooks implementing the entire distillation, fine-tuning, and deployment process

- **LocalFoundry Env/**: Contains Local Foundry model configuration file needed for local onnx inference in edge devices
  

## Workflow Introduction

The workshop flow is divided into the following key steps:

### 1. Knowledge Distillation (Implemented via DeepSeek)
In `01.AzureML_DistrillationByDeepSeek.ipynb`, we demonstrate how to:
- Load a dataset from Hugging Face datasets(in this case, a commonsense QA dataset)
- Prepare data for knowledge distillation
- Use a "teacher" model (a larger pre-trained model) to generate high-quality answers
- These answers will be used to train a "student" model (phi-4-mini-instruct)

### 2. Model Fine-tuning and Conversion (Via Microsoft Olive)
In `02.AzureML_FineTuningAndConvertByMSOlive.ipynb`, we:
- Fine-tune the Phi-4-mini model using the LoRA (Low-Rank Adaptation) method
- Train the model using data from the knowledge distillation phase
- Use Microsoft Olive tools to optimize and convert the model to ONNX format
- Apply quantization techniques (reduced to int4 precision) to decrease model size and improve inference efficiency

### 3. Model Inference Using ONNX Runtime GenAI
In `03.AzureML_RuningByORTGenAI.ipynb`, we show:
- How to load the optimized model in ONNX format
- Configure adapters and tokenizers
- Set up inference parameters
- Perform actual inference and generate responses using the fine-tuned model

### 4. Model Registration to AzureML
In `04.AzureML_RegisterToAzureML.ipynb`, we:
- Register the optimized model to the Azure Machine Learning workspace
- Set appropriate model metadata and descriptions
- Prepare the model for subsequent deployment or sharing

### 5. Local Model Download
In `05.Local_Download.ipynb`, we:
- Query and download registered models from the AzureML workspace
- Prepare model files for local development or deployment

## Technology Stack

The project uses multiple advanced AI and cloud technologies:

- **Azure Machine Learning**: For managing model training, registration, and deployment (https://ml.azure.com)
- **Microsoft Olive**: For model optimization and conversion(https://github.com/microsoft/olive)
- **ONNX Runtime GenAI**: For efficient model inference(https://github.com/microsoft/onnxruntime-genai)
- **Phi-4-mini**: Microsoft's SLM(https://huggingface.co/microsoft/Phi-4-mini-instruct)
- **LoRA**: An efficient model fine-tuning technique(https://learn.microsoft.com/en-us/azure/aks/concepts-fine-tune-language-models)
- **Foundry Local**: AI Foundry Local brings the power of Azure AI Foundry to your local devic(https://github.com/kinfey/Foundry-Local/blob/main/docs/how-to/compile-models-for-foundry-local.md)

## Usage Guide

1. Execute the notebook files in sequence (01 through 05)
    - AzureML
        - Please run `01.AzureML_DistrillationByDeepSeek.ipynb` and `04.AzureML_RegisterToAzureML.ipynb` in 'Python 3.10 AzureML' env
        - Please run `02.AzureML_FineTuningAndConvertByMSOlive.ipynb` and `03.AzureML_RuningByORTGenAI.ipynb` in 'Python 3.10 PyTorch and Tensorflow' env 
    - Local
        - Please run `05.Local_Download.ipynb` in your local env( Please install Python 3.10+ in your edge device)
2. Ensure required environment variables are set before execution , add .env in Notebook folder
```
TEACHER_MODEL_NAME = "DeepSeek-V3"
TEACHER_MODEL_ENDPOINT = "Your Azure AI Foundry DeepSeek-V3 Endpoint"
TEACHER_MODEL_KEY = "Your Azure AI Foundry DeepSeek-V3 Key"

AZUREML_WS_NAME = "Your Azure ML Workspace Name"
AZUREML_RESOURCE_GROUP = "Your Azure ML Resource Group Name"
AZUREML_SUBSCRIPTION_ID = "Your Azure Subscription ID"
```
3. Azure subscription and appropriate permissions are needed to use AzureML services
4. Local execution requires sufficient computing resources (especially for model fine-tuning steps) - A100 as your GPU 
5. Confirm you have installed Foundry Local sdk in your edge device
6. After finish `05.Local_Download.ipynb`, please copy `LocalFoundryEnv/inference_model.json` to your download folder , such as 'ft-phi-4-onnx-int4-cpu/onnx'

```bash
foundry cache cd model
 
foundry model run model
```

## Application Scenarios

The techniques demonstrated in this workshop are applicable to:
- Transferring knowledge from large language models to resource-constrained devices
- Optimizing models to reduce deployment and operational costs
- Customizing language models for specific domains or tasks
- Establishing end-to-end AI model optimization and deployment workflows

Through this project, you will learn how to leverage Azure AI Foundry capabilities to optimize and deploy efficient AI models while maintaining performance comparable to larger models.

