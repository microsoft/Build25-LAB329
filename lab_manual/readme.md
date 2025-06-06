## Welcome to Model Distillation with Microsoft Azure AI Foundry Workshop - Build 2025

This lab manual provides comprehensive guidance for the Model Distillation workshop using Microsoft Azure AI Foundry and advanced optimization tools. The workshop demonstrates a complete end-to-end workflow for distilling knowledge from Large Language Models to smaller, more efficient models.

### Workshop Structure

This lab is organized into five key sections following a practical workflow:

1. **Knowledge Distillation (AzureML & MAI)** - Using a teacher model to generate high-quality training data
2. **Fine-tuning and Model Conversion (Microsoft Olive)** - Fine-tuning with LoRA and optimizing with int4 quantization
3. **Model Inference (ONNX Runtime GenAI)** - Running the optimized model efficiently
4. **Model Registration (AzureML)** - Registering models in Azure ML for reuse
5. **Local Model Download** - Preparing models for edge deployment

### Time Allocation
## Workshop Duration and Timing
- **Total Workshop Time**: 70 minutes
- **Setup Time**: 5 minutes
- **Hands-On Activities**: 60 minutes
- **Discussion Time**: 5 minutes

### Workshop Materials

- **Interactive Notebooks**: Five Jupyter notebooks with detailed comments guide you through each step
- **Overview Documents**: Each notebook has a companion overview document explaining key concepts

### Technology Stack

This workshop leverages several cutting-edge technologies:

- **Azure Machine Learning**: For managing the end-to-end ML lifecycle requring a A100 Compute Instance
- **Microsoft Olive**: For model optimization and ONNX conversion
- **ONNX Runtime GenAI**: For efficient model inference
- **Phi-4-mini**: Microsoft's small but powerful language model
- **LoRA**: Low-Rank Adaptation for efficient fine-tuning
- **AI Foundry Local**: For local deployment and testing

Each notebook includes detailed explanations, code samples, and visualization of results to help participants understand both the theoretical concepts and practical implementation of model distillation techniques.

### Workshop Goals

By the end of this workshop, participants will be able to:
1. Implement knowledge distillation using Microsoft Azure AI Foundry
2. Optimize models using Microsoft Olive with int4 quantization
3. Run inference efficiently using ONNX Runtime GenAI
4. Deploy optimized models to both cloud and edge environments
5. Understand the trade-offs between model size, speed, and accuracy

### Setup instructions 

Please follow the [setup instructions](./setup_instructions.md) to configure you Azure AI Foundry Workspace and Azure ML Compute and Notebook environment.

### Lab Manual 
# Model Distillation Lab Manual: Teaching Small Models to Be Smart

## Duration
- Total Time: 70 minutes
- Reading Time: 20 minutes
- Hands-On Activities: 50 minutes

[Lab Manual Getting Started](./distillation_lab_manual.md)