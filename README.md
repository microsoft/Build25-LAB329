<p align="center">
<img src="img/banner.jpg" alt="decorative banner" width="1200"/>
</p>

# Build 2025 Lab329 

## Fine-Tune End-to-End Distillation Models with Azure AI Foundry Models
This workshop provides an in-depth journey into fine-tuning an end-to-end distillation process utilizing DeepSeek V3 as a teacher model and Phi4-mini as a student model. Participants will explore the theoretical underpinnings, practical applications, and engage in hands-on exercises to perfect the art of implementing and optimizing distillation techniques in AI projects.

Key topics include the concept of model distillation and its significance in modern AI, an overview of DeepSeek V3 and Phi4-mini, step-by-step demonstrations of the fine-tuning process, real-world case studies, and discussions of best practices and optimization strategies. Attendees will gain insights into leveraging the Azure AI Foundry, and the Azure AI Foundry Models to streamline model selection, enhance fine-tuning efficiency, and optimize deployment strategies.
Tailored for data scientists, machine learning engineers, and AI enthusiasts, this session equips attendees with critical skills to elevate their AI solutions through advanced distillation techniques and Azure-powered tooling.

This workshop provides hands-on experience with model distillation using Microsoft Azure AI Foundry. Learn how to extract knowledge from Large Language Models (LLMs) and transfer it to Smaller Language Models (SLMs) while maintaining good performance.

## Workshop Overview

Through a series of notebooks, this workshop demonstrates the complete workflow of model distillation, fine-tuning, and deployment using Azure Machine Learning (AzureML) platform, with a particular focus on optimizing models and deploying them to production environments.

### Folder Structure

- **Lab329/**: Main workshop content
  - **Notebooks/**: Jupyter notebooks implementing the entire distillation process
  - **LocalFoundryEnv/**: Configuration files for local ONNX inference on edge devices
- **lab_manual/**: Detailed lab manual with step-by-step instructions

### Workshop Flow

The workshop follows these key steps:

1. **Knowledge Distillation** ([`01.AzureML_Distillation.ipynb`](./Lab329/Notebook/01.AzureML_Distillation.ipynb)):
   - Load a commonsense QA dataset from Hugging Face
   - Prepare data for knowledge distillation
   - Use a "teacher" model to generate high-quality answers for training the "student" model
   - [Overview](./Lab329/Notebook/01.Overview.md)

2. **Model Fine-tuning and Conversion** ([`02.AzureML_FineTuningAndConvertByMSOlive.ipynb`](./Lab329/Notebook/02.AzureML_FineTuningAndConvertByMSOlive.ipynb)):
   - Fine-tune the Phi-4-mini model using the LoRA (Low-Rank Adaptation) method
   - Use Microsoft Olive tools to optimize and convert the model to ONNX format
   - Apply quantization techniques (int4 precision) to decrease model size
   - [Overview](./Lab329/Notebook/02.Overview.md)

3. **Model Inference Using ONNX Runtime GenAI** ([`03.AzureML_RuningByORTGenAI.ipynb`](./Lab329/Notebook/03.AzureML_RuningByORTGenAI.ipynb)):
   - Load the optimized model in ONNX format
   - Configure adapters and tokenizers
   - Perform inference and generate responses
   - [Overview](./Lab329/Notebook/03.Overview.md)

4. **Model Registration to AzureML** ([`04.AzureML_RegisterToAzureML.ipynb`](./Lab329/Notebook/04.AzureML_RegisterToAzureML.ipynb)):
   - Register the optimized model to the Azure Machine Learning workspace
   - Set appropriate model metadata for deployment
   - [Overview](./Lab329/Notebook/04.Overview.md)

5. **Local Model Download** ([`05.Local_Download.ipynb`](./Lab329/Notebook/05.Local_Download.ipynb)):
   - Download registered models for local development or deployment
   - [Overview](./Lab329/Notebook/05.Overview.md)

## Session Resources 

| Resources          | Links                             | Description        |
|:-------------------|:----------------------------------|:-------------------|
| Build session page | https://build.microsoft.com/sessions/SESSIONCODE | Event session page with downloadable recording, slides, resources, and speaker bio |

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
