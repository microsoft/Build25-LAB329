# **Model Distillation and Azure AI Foundry Workshop**

## Workshop Overview

This is a practical workshop on model distillation using Azure AI Foundry. Through a series of notebooks, it demonstrates how to extract knowledge from LLMs and transfer it to SLMs while maintaining good performance. This process utilizes various features of the Azure Machine Learning (AzureML) platform, with a particular focus on optimizing models and deploying them to production environments.

## Folder Structure

- **Notebooks /**: Contains Jupyter notebooks implementing the entire distillation, fine-tuning, and deployment process

- **LocalFoundry Env/**: Contains Local Foundry model configuration file needed for local onnx inference in edge devices
  
## Workflow Introduction

## Scenario

**Scenario: Edge AI for Education — Efficient Question Answering on Resource-Constrained Devices**

Imagine you are an AI engineer at an EdTech company. Your goal is to deliver an intelligent question-answering assistant that can run on affordable, resource-constrained devices in schools, such as laptops or edge servers, without requiring constant cloud connectivity. However, the best-performing language models are large and expensive to run locally.

In this workshop, you will:

- Use a large, cloud-hosted language model (the "teacher") to generate high-quality answers for a multiple-choice question dataset (CommonsenseQA).
- Distill this knowledge into a much smaller, efficient "student" model (Phi-4-mini) using knowledge distillation and fine-tuning techniques.
- Optimize and quantize the student model (using Microsoft Olive and ONNX) so it can run efficiently on local hardware.
- Register, download, and deploy the optimized model to an edge device using Azure AI Foundry Local.
- Validate that your compact model can answer questions with high accuracy and low latency, even on limited hardware.

This scenario reflects real-world needs for deploying AI in education, healthcare, manufacturing, and other sectors where cost, privacy, and offline capability are critical. By the end of the lab, you will have built an end-to-end workflow for distilling, optimizing, and deploying advanced AI models from the cloud to the edge.

## The workshop flow is divided into the following key steps

**Notebook** 01.AzureML_Distillation

- **Purpose:** Generate training data using DeepSeek-V3  
- **Time:** 15 min  

**Notebook** 02.AzureML_FineTuningAndConvertByMSOlive

- **Purpose:** Fine-tune Phi-4-mini with LoRA and optimize  
- **Time:** 15 min  

**Notebook** 03.AzureML_RuningByORTGenAI  

- **Purpose:** Test model inference with ONNX Runtime  
- **Time:** 10 min  

**Notebook** 04.AzureML_RegisterToAzureML  

- **Purpose:** Register model to Azure ML  
- **Time:** 5 min  

**Notebook** 05.Local_Download  

- **Purpose:** Download model for local deployment  
- **Time:** 5 min  

**Notebook** 06.Local_Inference  

- **Purpose:** Run inference locally  
- **Time:** 10 min  

**Notebook** 07.Local_inference_AIFoundry  

- **Purpose:** Run inference locally with Foundry Local  
- **Time:** 10 min  

## Technology Stack

The project uses multiple advanced AI and cloud technologies:

- **Azure Machine Learning**: For managing model training, registration, and deployment (<https://ml.azure.com>)
- **Microsoft Olive**: For model optimization and conversion(<https://github.com/microsoft/olive>)
- **ONNX Runtime GenAI**: For efficient model inference(<https://github.com/microsoft/onnxruntime-genai>)
- **Phi-4-mini**: Microsoft's SLM(<https://huggingface.co/microsoft/Phi-4-mini-instruct>)
- **LoRA**: An efficient model fine-tuning technique(<https://learn.microsoft.com/en-us/azure/aks/concepts-fine-tune-language-models>)
- **Foundry Local**: AI Foundry Local brings the power of Azure AI Foundry to your local device(<https://github.com/microsoft/Foundry-Local/blob/main/docs/how-to/compile-models-for-foundry-local.md>)

## Usage Guide **Please ensure you run this in the Azure ML Notebook Workspace and select the correct Kernel**

1. Execute the notebook files in sequence (01 through 05) using [Azure ML Workspace](https://ml.azure.com)
    - AzureML
        - Please run `01.AzureML_Distillation.ipynb` and `04.AzureML_RegisterToAzureML.ipynb` in 'Python 3.10 AzureML' env
        - Please run `02.AzureML_FineTuningAndConvertByMSOlive.ipynb` and `03.AzureML_RuningByORTGenAI.ipynb` in 'Python 3.10 PyTorch and Tensorflow' env
    - Local
        - Please run `05.Local_Download.ipynb` in your local env( Please install Python 3.10+ in your edge device)
2. Ensure required environment variables are set before execution , add local.env in Notebook folder

```
TEACHER_MODEL_NAME=your-model-name
TEACHER_MODEL_ENDPOINT=https://your-endpoint.services.ai.azure.com/models
TEACHER_MODEL_KEY=your-api-key-here

# Azure ML workspace information
AZUREML_SUBSCRIPTION_ID=your-subscription-id
AZUREML_RESOURCE_GROUP=your-resource-group
AZUREML_WS_NAME=your-workspace-name
```

3. Azure subscription and appropriate permissions are needed to use AzureML services see [Setup Instructions](../lab_manual/setup_instructions.md)
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
