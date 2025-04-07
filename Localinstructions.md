## Local A100 GPU Implementation Overview

- local_gpu_train.py: A script specifically optimized for training on A100 GPUs with features like:
- Mixed precision training (FP16) for improved performance
- Multi-GPU support with Distributed Data Parallel (DDP)
- Gradient accumulation for training with larger batch sizes
- Automatic detection of available GPUs
generate_local_data.py: A script to generate training data locally using OpenAI models without requiring Azure

Requirements:
- Azure [A100 Compute Instance with GPU support](https://learn.microsoft.com/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series?tabs=sizebasic)
- Python 3.8 or higher
- PyTorch 1.9 or higher with CUDA support
- Transformers library from Hugging Face
- OpenAI API key for data generation (if using OpenAI models)

## Edit Requirememnts.txt to support Local Deploment
requirements.txt: Added necessary dependencies for local A100 GPU training ensure you uncomment libraries required

## Unit tests: 
Tests for the local GPU implementation

## How to Use the Local A100 GPU Implementation

## Step 1: Set up Environment
First, install the required dependencies:
```
pip install -r requirements.txt
```
### Step 2: Generate Training Data
You'll need to generate training data first:
```
python src/generate_local_data.py --num_examples 1000 --output_path ./data/distillation_data.jsonl --model gpt-4o
```
Options:

- num_examples: Number of training examples to generate (default: 100)
- output_path: Where to save the generated data (default: ./data/distillation_data.jsonl)
- model: Which model to use as teacher (default: gpt-4o)
- max_tokens: Maximum tokens per response (default: 200)
- temperature: Sampling temperature (default: 0.7)
- api_key_env_var: Environment variable name containing your OpenAI API key (default: OPENAI_API_KEY)

### Step 3: Train on A100 GPU
Now you can run the training on your A100 GPU:
```
python src/local_gpu_train.py --dataset_path ./data/distillation_data.jsonl --student_model distilgpt2 --output_dir ./outputs --fp16
```

### Key options:

- dataset_path: Path to your generated training data (required)
- student_model: Which model architecture to use as student (default: distilgpt2)
- output_dir: Directory to save model checkpoints and final model (default: ./outputs)
- fp16: Enable mixed precision training for better performance on A100 GPUs
- batch_size: Training batch size (default: from config)
- gradient_accumulation_steps: Accumulate gradients over multiple batches (default: 1)
The script automatically detects all available GPUs and uses them for training with distributed data parallel.

### A100-Specific Optimizations
The implementation includes several optimizations specifically for A100 GPUs:

- Mixed Precision Training: The --fp16 flag enables FP16 training with automatic mixed precision, which can provide up to 2-3x speedup on A100 GPUs.
- Distributed Training: If you have multiple A100 GPUs, the code automatically distributes training across all of them, synchronizing gradients for better convergence.
- Gradient Accumulation: For training with larger effective batch sizes even with limited GPU memory.
- Memory Efficiency: The implementation uses best practices for memory management to maximize the model size you can train on your A100.

### Monitoring Training
During training, you'll see:

- Loss values for each batch and epoch
- GPU utilization information
- Progress bars for each epoch
- Model checkpoints saved after each epoch

The final model will be saved to [output_dir]/final_model/ and can be loaded using the Hugging Face Transformers library.