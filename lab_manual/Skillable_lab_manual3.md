### Step 3: Test Your ONNX Model (10 min)

**Notebook:** `03.AzureML_RuningByORTGenAI.ipynb`

**Purpose:** Test the optimized model using ONNX Runtime GenAI to verify its performance on multiple-choice questions.

#### Instructions:

1. **Open the notebook** from the file explorer

# Model Inference with ONNX Runtime GenAI

This notebook `03.AzureML_RuningByORTGenAI.ipynb` implements the third phase of our model distillation pipeline: running inference with the optimized ONNX model using ONNX Runtime GenAI. This demonstrates how the fine-tuned and optimized model performs on our target task.

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

## Technical Components

### Environment Setup
- Installation of `onnxruntime-genai` package for inference with optimized models
- Setting up necessary imports for NumPy and OS operations

### Model Loading
- Loading the ONNX model from the output directory of the previous notebook
- Using `og.Model()` to initialize the model
- Setting up adapters to integrate the fine-tuned weights with the base model

### Tokenizer Configuration
- Creating a tokenizer instance for the model
- Initializing a tokenizer stream for efficient token processing
- Setting up a chat template to properly format inputs

### Inference Parameters
- Configuring generation parameters:
  - `max_length`: Setting maximum generation length
  - `past_present_share_buffer`: Memory optimization option
- Defining the proper prompt structure with system instructions

### Sample Inference
- Preparing sample question input
- Tokenizing the input with appropriate chat formatting
- Running the generator with the correct adapter
- Processing and displaying the model's output

## Code Highlights

```python
# Load the ONNX model and adapter
model = og.Model(model_folder)
adapters = og.Adapters(model)
adapters.load('./models/phi-4-mini/onnx/model/adapter_weights.onnx_adapter', "qa_choice")

# Set up tokenizer
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Format input with chat template
chat_template = "</s>You are a helpful assistant. Your output should only be one of the five choices: 'A', 'B', 'C', 'D', or 'E'.<|end|><|user|>{input}<|end|><|assistant|>"
prompt = f'{chat_template.format(input=input)}'
input_tokens = tokenizer.encode(prompt)

# Configure inference parameters
params = og.GeneratorParams(model)
params.set_search_options(**search_options)
generator = og.Generator(model, params)
generator.set_active_adapter(adapters, "qa_choice")

# Run inference
generator.append_tokens(input_tokens)
```

## Benefits of This Approach

1. **Efficiency**: The ONNX Runtime provides optimized inference for the quantized model
2. **Adapter Integration**: Seamlessly integrates the fine-tuned adapter with the base model
3. **Format Preservation**: Maintains the same input/output format used during fine-tuning
4. **Deployment Readiness**: Demonstrates the model in a format ready for production deployment
5. **Validation**: Allows immediate validation of the distillation and optimization process

This notebook demonstrates the practical value of the distillation and optimization pipeline by showing the model in action. The efficient inference capabilities of ONNX Runtime combined with the knowledge distilled from the teacher model result in a high-performance, compact model suitable for deployment in resource-constrained environments.
