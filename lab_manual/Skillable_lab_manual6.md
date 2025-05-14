### Step 6: Local Inference (10 min)

**Notebook:** `06.Local_Inference.ipynb`

**Purpose:** Run the optimized model on your local machine to demonstrate its ability to answer questions without cloud resources. You will be running this notebook from the skillable VM. Please open the folder `C:\Users\LabUser\Desktop\lab\Build25-LAB329\Lab329\Notebook` which is located on the desktop of your VM called `lab`.

We have also provided a downloaded version of the model in the `C:\Users\LabUser\Desktop\lab\fine-tuning-phi-4-mini-onnx-int4-cpu` this can be accessed via the `lab` folder on the desktop.

# Local Model Inference

This document explains the process of running inference with your distilled, optimized model on local hardware using ONNX Runtime GenAI.

## Purpose

After downloading the distilled model from Azure ML, this notebook demonstrates how to:
1. Load the model locally with ONNX Runtime GenAI
2. Set up a tokenizer and generator for text generation
3. Process multiple-choice questions
4. Generate accurate responses without cloud resources

## Technical Details

### ONNX Runtime GenAI

ONNX Runtime GenAI is a specialized runtime for generative AI models that provides:
- Optimized inference for transformer-based models
- Support for adapter-based fine-tuning
- Cross-platform compatibility for various hardware targets
- Efficient memory management for resource-constrained environments

### Model Loading and Adapters

The notebook demonstrates:
- Loading the quantized ONNX model from a local path
- Attaching fine-tuning adapters that contain the task-specific knowledge
- Configuring generation parameters for optimal performance
- Setting up a proper tokenization pipeline

### Inference Process

The inference workflow follows these steps:
1. Format the input question with appropriate prompting
2. Tokenize the input using the model's tokenizer
3. Configure generation parameters (temperature, max length, etc.)
4. Generate tokens using the ONNX Runtime GenAI generator
5. Process the output to extract the relevant answer choice

## Benefits of Local Inference

Running inference locally provides several advantages:
- **Privacy**: All data stays on your device
- **Offline capability**: No internet connection required
- **Reduced latency**: No network roundtrips to cloud services
- **No API costs**: Free to run as many queries as needed
- **Deployment flexibility**: Can be integrated directly into applications

This notebook completes the end-to-end distillation workflow by demonstrating that the optimized model can successfully run on local hardware while maintaining accuracy on the intended task.
