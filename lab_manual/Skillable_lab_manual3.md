
# Step 3: Test Your ONNX Model (10 min)

**Notebook:** `03.AzureML_RuningByORTGenAI.ipynb`

**Purpose:** Test the optimized model using ONNX Runtime GenAI to verify its performance on multiple-choice questions.

#### Instructions

1. **Open the notebook** from the file explorer.

---

# Model Inference with ONNX Runtime GenAI

This notebook (`03.AzureML_RuningByORTGenAI.ipynb`) implements the third phase of our model distillation pipeline: running inference with the optimized ONNX model using ONNX Runtime GenAI. This demonstrates how the fine-tuned and optimized model performs on our target task.

## Purpose

This notebook demonstrates practical inference with the distilled model by:
1. Loading the optimized ONNX model created in the previous step
2. Configuring the model with proper tokenization and adapters
3. Setting up inference parameters for optimal generation
4. Running inference on sample questions
5. Evaluating the model's performance

## Workflow Overview

1. **Environment Setup**: Installing the ONNX Runtime GenAI package
2. **Model Loading**: Loading the optimized ONNX model and adapter weights
3. **Tokenizer Configuration**: Setting up the tokenizer for input processing
4. **Inference Parameters**: Configuring generation parameters
5. **Sample Inference**: Running inference on example questions


## Benefits of This Approach

1. **Efficiency**: The ONNX Runtime provides optimized inference for the quantized model
2. **Adapter Integration**: Seamlessly integrates the fine-tuned adapter with the base model
3. **Format Preservation**: Maintains the same input/output format used during fine-tuning
4. **Deployment Readiness**: Demonstrates the model in a format ready for production deployment
5. **Validation**: Allows immediate validation of the distillation and optimization process

This notebook demonstrates the practical value of the distillation and optimization pipeline by showing the model in action. The efficient inference capabilities of ONNX Runtime combined with the knowledge distilled from the teacher model result in a high-performance, compact model suitable for deployment in resource-constrained environments.
