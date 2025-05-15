
# Step 2: Fine-tune and Optimize (15 min)

**Notebook:** `02.AzureML_FineTuningAndConvertByMSOlive.ipynb`

**Purpose:** Transform the small student model by fine-tuning it on the training data generated from the teacher model, and optimize it for deployment.

#### Instructions

1. **Open the notebook** from the file explorer in Azure ML Studio.

---

# Fine-Tuning and Model Conversion with Microsoft Olive

This notebook (`02.AzureML_FineTuningAndConvertByMSOlive.ipynb`) implements the second critical phase in our model distillation pipeline: fine-tuning the student model with the knowledge captured from the teacher model, and then optimizing it for efficient deployment.

## Workflow Overview

1. **Environment Setup**: Installation of essential libraries for model fine-tuning and optimization
2. **Fine-Tuning with LoRA**: Parameter-efficient tuning of Phi-4-mini-instruct using distilled knowledge
3. **Model Optimization**: Converting and optimizing the model to ONNX format with int4 quantization

## Benefits of This Approach

1. **Efficiency**: LoRA fine-tuning is significantly faster and requires less computational resources than full model fine-tuning
2. **Knowledge Transfer**: Successfully transfers knowledge from the larger teacher model to the smaller student model
3. **Deployment Optimization**: Int4 quantization drastically reduces model size, enabling deployment on edge devices
4. **Performance**: Despite the optimizations, the model maintains most of its performance on the target task

## Next Steps

After completing this notebook:
1. The fine-tuned and optimized model is ready for inference (demonstrated in notebook 03)
2. The ONNX format enables efficient deployment across various platforms
3. The int4 quantization makes the model suitable for resource-constrained environments

This technique demonstrates how Microsoft's Olive toolkit simplifies the complex process of fine-tuning and optimizing transformer models while significantly improving their deployment characteristics.
