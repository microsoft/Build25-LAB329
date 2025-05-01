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


## The workshop flow is divided into the following key steps:

### 1. Knowledge Distillation (Implemented via DeepSeek)
In `01.AzureML_Distillation.ipynb`, we demonstrate how to:
- Load a dataset from Hugging Face datasets(in this case, a commonsense QA dataset)
- Prepare data for knowledge distillation
- Use a "teacher" model (a larger pre-trained model) to generate high-quality answers
- These answers will be used to train a "student" model (phi-4-mini-instruct)

Please read the Notebook 1 [Overview](./Notebook/01.Overview.md)

### 2. Model Fine-tuning and Conversion (Via Microsoft Olive)
In `02.AzureML_FineTuningAndConvertByMSOlive.ipynb`, we:
- Fine-tune the Phi-4-mini model using the LoRA (Low-Rank Adaptation) method
- Train the model using data from the knowledge distillation phase
- Use Microsoft Olive tools to optimize and convert the model to ONNX format
- Apply quantization techniques (reduced to int4 precision) to decrease model size and improve inference efficiency

Please read the Notebook 2 [Overview](./Notebook/02.Overview.md)

### 3. Model Inference Using ONNX Runtime GenAI
In `03.AzureML_RuningByORTGenAI.ipynb`, we show:
- How to load the optimized model in ONNX format
- Configure adapters and tokenizers
- Set up inference parameters
- Perform actual inference and generate responses using the fine-tuned model

Please read the Notebook 3 [Overview](./Notebook/03.Overview.md)

### 4. Model Registration to AzureML
In `04.AzureML_RegisterToAzureML.ipynb`, we:
- Register the optimized model to the Azure Machine Learning workspace
- Set appropriate model metadata and descriptions
- Prepare the model for subsequent deployment or sharing

Please read the Notebook 4 [Overview](./Notebook/04.Overview.md)

### 5. Local Model Download
In `05.Local_Download.ipynb`, we:
- Query and download registered models from the AzureML workspace
- Prepare model files for local development or deployment

Please read the Notebook 5 [Overview](./Notebook/05.Overview.md)

## Technology Stack

The project uses multiple advanced AI and cloud technologies:

- **Azure Machine Learning**: For managing model training, registration, and deployment (https://ml.azure.com)
- **Microsoft Olive**: For model optimization and conversion(https://github.com/microsoft/olive)
- **ONNX Runtime GenAI**: For efficient model inference(https://github.com/microsoft/onnxruntime-genai)
- **Phi-4-mini**: Microsoft's SLM(https://huggingface.co/microsoft/Phi-4-mini-instruct)
- **LoRA**: An efficient model fine-tuning technique(https://learn.microsoft.com/en-us/azure/aks/concepts-fine-tune-language-models)
- **Foundry Local**: AI Foundry Local brings the power of Azure AI Foundry to your local device(https://github.com/microsoft/Foundry-Local/blob/main/docs/how-to/compile-models-for-foundry-local.md)

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

