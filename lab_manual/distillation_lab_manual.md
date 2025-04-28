# Model Distillation Lab Manual: Teaching Small Models to Be Smart

## Workshop Duration and Timing
- **Total Workshop Time**: 70 minutes
- **Setup Time**: 5 minutes
- **Hands-On Activities**: 60 minutes
- **Discussion Time**: 5 minutes

## Table of Contents
- [Workshop Overview](#workshop-overview)
- [Prerequisites Check](#prerequisites-check)
- [Environment Setup](#environment-setup)
- [Notebook-by-Notebook Guide](#notebook-by-notebook-guide)
  - [Step 1: Generate Training Data (15 min)](#step-1-generate-training-data-15-min)
  - [Step 2: Fine-tune and Optimize (15 min)](#step-2-fine-tune-and-optimize-15-min)
  - [Step 3: Test Your ONNX Model (10 min)](#step-3-test-your-onnx-model-10-min)
  - [Step 4: Register to Azure ML (5 min)](#step-4-register-to-azure-ml-5-min)
  - [Step 5: Download the Model (5 min)](#step-5-download-the-model-5-min)
  - [Step 6: Local Inference (10 min)](#step-6-local-inference-10-min)
- [Troubleshooting Guide](#troubleshooting-guide)
- [What You've Learned](#what-youve-learned)
- [Next Steps](#next-steps)

## Workshop Overview

Welcome to the Model Distillation Workshop! In this hands-on session, you'll learn how to transform a large language model (DeepSeek-V3) into a smaller, equally capable model (Phi-4-mini) using knowledge distillation.

### What You'll Build

By the end of this workshop, you'll create a compact language model that:
- Is **75% smaller** than the original teacher model
- Runs on **standard hardware** without specialized GPUs
- Can be deployed on **edge devices** or embedded systems
- Maintains most of the **capabilities** of the larger model

### Workshop Flow

This is a **practical, code-first workshop**. You'll work through six Jupyter notebooks that guide you step-by-step through:

1. **Data Generation**: Create training examples using a large "teacher" model
2. **Fine-Tuning**: Train a smaller "student" model on this data
3. **Optimization**: Convert and compress the model for efficiency
4. **Testing**: Verify the model works correctly
5. **Registration**: Register your model with Azure ML
6. **Deployment**: Download and run the model locally

Each notebook is designed to be completed in 5-15 minutes, with clear instructions at each step.

## Prerequisites Check

Before you start, ensure you have:

- [x] **Azure ML Studio access** with compute resources allocated
- [x] **Python 3.10+** installed (already in Azure ML Studio)
- [x] **Basic Python knowledge** (understanding of functions, loops, and imports)
- [x] **Terminal/command line familiarity** (basic git commands)
- [x] **Internet connectivity** (to access datasets and models)

> **Important:** This workshop uses Azure ML Studio and requires access to a deployed Azure AI model endpoint. You should have already been provided with the necessary credentials.

## Environment Setup

Let's start by setting up your environment and cloning the code repository.

### 1. Access Azure ML Studio

Open Azure ML Studio by:
1. Go to [Azure ML Studio](https://ml.azure.com/)
2. Sign in with your Azure credentials
3. Select your workspace (provided by your instructor)
4. Navigate to the "Notebooks" section

### 2. Clone the Repository

1. In Azure ML Studio, click on the terminal icon at the top right
2. Run these commands to clone and navigate to the repo:

```bash
git clone https://github.com/microsoft/Build25-LAB329
cd Build25-LAB329/Lab329/Notebook
```

### 3. Create Your Environment File

Create a local environment file to store your Azure credentials:

```bash
cp sample.env local.env
code local.env
```

Add your credentials to the file (these will be provided by your instructor):

```
TEACHER_MODEL_NAME=DeepSeek-V3
TEACHER_MODEL_ENDPOINT=https://your-endpoint.services.ai.azure.com/models
TEACHER_MODEL_KEY=your-api-key-here
AZUREML_SUBSCRIPTION_ID=your-subscription-id
AZUREML_RESOURCE_GROUP=your-resource-group
AZUREML_WS_NAME=your-workspace-name
```

Save the file and close the editor.

## Notebook-by-Notebook Guide

This workshop uses 6 Jupyter notebooks that you'll run in sequence. Each notebook builds on the previous one, so it's important to complete them in order.

Let's look at each notebook and what you'll do:

| Notebook | Purpose | Duration |
|----------|---------|----------|
| 01_AzureML_Distillation | Generate training data using DeepSeek-V3 | 15 min |
| 02_AzureML_FineTuningAndConvertByMSOlive | Fine-tune Phi-4-mini with LoRA and optimize | 15 min |
| 03_AzureML_RuningByORTGenAI | Test model inference with ONNX Runtime | 10 min |
| 04_AzureML_RegisterToAzureML | Register model to Azure ML | 5 min |
| 05_Local_Download | Download model for local deployment | 5 min |
| 06_Local_Inference | Run inference locally | 10 min |

Let's start with the first notebook!

## Clone the GitHub Repo and resources to your Azure ML Studio 

Open your [Azure ML Studio](https://ml.azure.com)

![MLStudio](./images/ML_Studio.png)

Select Notebooks

![MLStudioNotebooks](./images/Notebook_Terminal.png)

Select Terminal

![SelectTerminal](./images/ML_Terminal.png)

To clone the repository and set up your environment for the lab, follow these steps:

1. **Clone the Repository**: Use the `git clone` command followed by the repository URL:
   ```bash
   git clone https://github.com/microsoft/Build25-LAB329 
   ```

2. **Access the Cloned Repository**: Navigate to the directory of the cloned repository:
   ```bash
   cd Build25-LAB329
   ```

3. **Navigate to the Lab Directory**: Go to the Lab329 folder containing the notebooks:
   ```bash
   cd Lab329/Notebook
   ```

4. **Create Your Environment File**: Copy the sample environment file and rename it to local.env:
   ```bash
   cp sample.env local.env
   ```

5. **Edit Your Environment File**: Update the local.env file with your Azure credentials using a text editor:
   ```bash
   code local.env
   ```

Login with your azure creditional 

```
az login --identity
```

Now you’re ready to work with the repository on your Azure ML Studio!


## Step 1: Generate Training Data (15 min)

**Notebook:** `01.AzureML_Distillation.ipynb`

**Purpose:** Create a training dataset by asking a large teacher model (DeepSeek-V3) to answer multiple-choice questions.

#### Instructions:

1. **Open the notebook** from the left file explorer panel in Azure ML Studio

2. **Read the purpose and overview** at the top of the notebook 

3. **Install required packages**
   - Run the first cell to install dependencies
   - Look for successful installation messages before continuing

4. **Load environment variables**
   - Run the environment variables cell
   - Verify your teacher model endpoint is correctly loaded

5. **Load the dataset**
   - Run the cell to load the CommonSenseQA dataset
   - You should see output showing the dataset was successfully loaded

6. **Process questions and get teacher responses**
   - Run the cells that process questions for the teacher model
   - Watch the progress as the teacher model generates answers
   - Note: This may take 5-10 minutes depending on the number of questions

7. **Save the teacher's responses**
   - Run the final cells to save responses to `data/train_data.jsonl`
   - Verify the file was created successfully

#### Key Outputs:
- A JSONL file with questions and expert answers from the teacher model
- This file will be used to train your student model in the next notebook


## Troubleshooting Guide

If you encounter issues during the workshop, use this guide to resolve common problems:

### Environment and Setup Issues

1. **Authentication errors**:
   - Error: "Failed to authenticate to Azure"
   - Solution: Verify your local.env file has the correct values
   - Fix: Run `az login` in the terminal to refresh your login

2. **Missing environment variables**:
   - Error: "NameError: name 'xyz' is not defined"
   - Solution: Make sure you've run all initialization cells
   - Fix: Create or update your local.env file with the required values

3. **Package installation failures**:
   - Error: "ERROR: Could not install packages..."
   - Solution: Ensure you have internet connectivity and proper permissions
   - Fix: Try installing one package at a time or specify --user flag

### Teacher Model Issues

1. **Connection to teacher model fails**:
   - Error: "Unable to connect to endpoint"
   - Solution: Check your API key and endpoint URL
   - Fix: Update the TEACHER_MODEL_* values in local.env

2. **Dataset loading errors**:
   - Error: "Failed to download/load dataset"
   - Solution: Check internet connectivity and reduce batch size
   - Fix: Try `dataset = load_dataset("tau/commonsense_qa", split="train[:10]")` for a smaller sample


### Step 1: Generate Training Data from a Teacher Model (15 minutes)

Open the [`01.AzureML_Distillation.ipynb`](../Lab329/Notebook/01.AzureML_Distillation.ipynb) notebook and follow these steps to generate training data for your student model:

1. **Run the notebook cell by cell**, starting with package installation:
   ```python
   pip install python-dotenv
   pip install datasets -U
   pip install azure-ai-inference
   ```

2. **Load and prepare a dataset**:
   ```python
   from datasets import load_dataset
   from abc import ABC
   
   # Define a class to handle the input dataset
   class InputDataset(ABC):
       def __init__(self):
           super().__init__()
           (
               self.train_data_file_name,
               self.test_data_file_name,
               self.eval_data_file_name,
           ) = (None, None, None)
   
   # Specific implementation for the QA dataset
   class CQnAHuggingFaceInputDataset(InputDataset):
       def __init__(self):
           super().__init__()
   
       def load_hf_dataset(
           self,
           dataset_name,
           train_sample_size=10,
           val_sample_size=10,
           test_sample_size=10,
           train_split_name="train",
           val_split_name="validation",
           test_split_name="test",
       ):
           # Load dataset and create splits
           full_dataset = load_dataset(dataset_name)
           train_data = full_dataset[train_split_name].select(range(train_sample_size))
           val_data = full_dataset[val_split_name].select(range(val_sample_size))
           test_data = full_dataset[test_split_name].select(range(test_sample_size))
           return train_data, val_data, test_data
   ```

3. **Sample data from a Hugging Face dataset**:
   ```python
   # Define sample sizes
   train_sample_size = 100
   val_sample_size = 100
   
   # We'll use the commonsense QA dataset
   dataset_name = "tau/commonsense_qa"
   input_dataset = CQnAHuggingFaceInputDataset()
   
   # Load the dataset
   train, val, _ = input_dataset.load_hf_dataset(
       dataset_name=dataset_name,
       train_sample_size=train_sample_size,
       val_sample_size=val_sample_size,
       train_split_name="train",
       val_split_name="validation",
   )
   ```

4. **Format the questions for the teacher model**:
   ```python
   import json
   
   # Create directory for data
   ! mkdir -p data
   train_data_path = "data/train_original_data.jsonl"
   
   # Define prompts
   system_prompt = "You are a helpful assistant. Your output should only be one of the five choices: 'A', 'B', 'C', 'D', or 'E'."
   user_prompt_template = "Answer the following multiple-choice question by selecting the correct option.\n\nQuestion: {question}\nAnswer Choices:\n{answer_choices}"
   
   # Format each question
   for row in train:
       data = {"messages": []}
       data["messages"].append(
           {
               "role": "system",
               "content": system_prompt,
           }
       )
       question, choices = row["question"], row["choices"]
       labels, choice_list = choices["label"], choices["text"]
       answer_choices = [
           "({}) {}".format(labels[i], choice_list[i]) for i in range(len(labels))
       ]
       answer_choices = "\n".join(answer_choices)
       data["messages"].append(
           {
               "role": "user",
               "content": user_prompt_template.format(
                   question=question, answer_choices=answer_choices
               ),
           }
       )
       with open(train_data_path, "a") as f:
           f.write(json.dumps(data) + "\n")
   ```

5. **Load credentials and connect to the teacher model**:
   ```python
   from dotenv import load_dotenv
   import os
   from azure.ai.inference import ChatCompletionsClient
   from azure.ai.inference.models import SystemMessage, UserMessage
   from azure.core.credentials import AzureKeyCredential
   
   # Load environment variables
   load_dotenv()
   
   teacher_model_name = os.getenv('TEACHER_MODEL_NAME')
   teacher_model_endpoint_url = os.getenv('TEACHER_MODEL_ENDPOINT')
   teacher_model_api_key = os.getenv('TEACHER_MODEL_KEY')
   
   # Set up client
   endpoint = teacher_model_endpoint_url
   model_name = teacher_model_name
   key = teacher_model_api_key
   client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
   ```

6. **Generate responses using the teacher model**:
   ```python
   def process_question(question_data):
       try:
           messages = []
           for msg in question_data["messages"]:
               if msg["role"] == "system":
                   messages.append(SystemMessage(content=msg["content"]))
               elif msg["role"] == "user":
                   messages.append(UserMessage(content=msg["content"]))
   
           response = client.complete(
               messages=messages,
               model=model_name,
               max_tokens=100  # Short answers (A, B, C, D, or E)
           )
   
           return {
               "question": question_data["messages"][1]["content"],
               "response": response.choices[0].message.content,
               "full_response": response
           }
       except Exception as e:
           return {
               "question": question_data["messages"][1]["content"] if len(question_data["messages"]) > 1 else "Error",
               "response": f"Error: {str(e)}",
               "full_response": None
           }
   
   # Process all questions
   results = []
   with open(train_data_path, 'r', encoding='utf-8') as file:
       print(f"Processing questions from {train_data_path}")
       for i, line in enumerate(file):
           if line.strip():  # Skip empty lines
               try:
                   question_data = json.loads(line)
                   print(f"Processing question {i+1}...")
                   result = process_question(question_data)
                   results.append(result)
                   print(f"Question {i+1} response: {result['response']}")
               except Exception as e:
                   print(f"Error processing line {i+1}: {str(e)}")
   ```

7. **Save the teacher's responses for student model training**:
   ```python
   output_file_path = "./data/train_data.jsonl"
   with open(output_file_path, 'w', encoding='utf-8') as f:
       for result in results:
           # Create the simplified output format
           output_line = {
               "Question": result["question"],
               "Answer": result["response"]
           }
   
           # Write as JSONL (one JSON object per line)
           f.write(json.dumps(output_line, ensure_ascii=False) + '\n')
   ```

This process:
- Loads a multiple-choice question dataset from Hugging Face
- Formats each question with a clear system prompt and instruction
- Sends the questions to your teacher model (using Azure AI Foundry MAI endpoint)
- Collects the high-quality responses from the larger model
- Creates a training dataset that pairs questions with expert answers
- Formats the output for use in the next step of model distillation

The resulting `train_data.jsonl` file will be used in Step 2 to fine-tune your smaller student model.

### Step 2: Fine-tune and Optimize (15 min)

**Notebook:** `02.AzureML_FineTuningAndConvertByMSOlive.ipynb`

**Purpose:** Transform the small student model by fine-tuning it on the training data generated from the teacher model, and optimize it for deployment.

#### Instructions:

1. **Open the notebook** from the file explorer in Azure ML Studio

2. **Install required packages**
   - Run the package installation cell:
   ```python
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -U
   pip install olive-ai[auto-opt] -U
   pip install onnxruntime-genai==0.7.0 --pre
   pip install peft
   ```
   - Wait for all packages to install successfully

3. **Fine-tune with LoRA**
   - Run the cell containing the olive finetune command:
   ```python
   !olive finetune \
       --method lora \
       --model_name_or_path azureml://registries/azureml/models/Phi-4-mini-instruct/versions/1 \
       --trust_remote_code \
       --data_name json \
       --data_files ./data/train_data.jsonl \
       --text_template "<|user|>{Question}<|end|><|assistant|>{Answer}<|end|>" \
       --max_steps 100 \
       --output_path models/phi-4-mini/ft \
       --target_modules "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj" \
       --log_level 1
   ```
   - This will take approximately 5-10 minutes to complete
   - Watch the output for "Fine-tuning complete" message

4. **Convert to ONNX and quantize**
   - Run the model optimization cell:
   ```python
   !olive auto-opt \
       --model_name_or_path azureml://registries/azureml/models/Phi-4-mini-instruct/versions/1 \
       --adapter_path models/phi-4-mini/ft/adapter \
       --device cpu \
       --provider CPUExecutionProvider \
       --use_model_builder \
       --precision int4 \
       --output_path models/phi-4-mini/onnx \
       --log_level 1
   ```
   - This will take approximately 5-10 minutes
   - The optimization reduces model size by ~75%

5. **Verify the output directory**
   - Run the cell to list the files in the output directory
   - Ensure both model files and adapter files are present

#### Key Outputs:
- A LoRA adapter in `models/phi-4-mini/ft/adapter`
- A quantized ONNX model in `models/phi-4-mini/onnx/model`
- These files will be used for inference in the next notebook


### Step 3: Test Your ONNX Model (10 min)

**Notebook:** `03.AzureML_RuningByORTGenAI.ipynb`

**Purpose:** Test the optimized model using ONNX Runtime GenAI to verify its performance on multiple-choice questions.

#### Instructions:

1. **Open the notebook** from the file explorer

2. **Import libraries and load the model**
   - Run the import and model loading cells:
   ```python
   import onnxruntime_genai as og
   import numpy as np
   
   model_folder = "./models/phi-4-mini/onnx/model"
   model = og.Model(model_folder)
   ```
   - Verify the model loads without errors

3. **Load the adapter**
   - Run the adapter loading cell:
   ```python
   adapters = og.Adapters(model)
   adapters.load('./models/phi-4-mini/onnx/model/adapter_weights.onnx_adapter', "qa_choice")
   ```
   - Check for confirmation that the adapter loaded successfully

4. **Set up the tokenizer**
   - Run the tokenizer setup cells:
   ```python
   tokenizer = og.Tokenizer(model)
   tokenizer_stream = tokenizer.create_stream()
   
   search_options = {}
   search_options['max_length'] = 102
   search_options['past_present_share_buffer'] = False
   search_options['repeat_penalty'] = 1.1
   search_options['temperature'] = 0.7
   ```

5. **Test the model on example questions**
   - Run the inference cells with example multiple-choice questions
   - The model should generate answers (A, B, C, D, or E)
   - Compare the model's answers to the expected answers

6. **Try your own questions** (optional)
   - Modify the example question in the last cell
   - Run the cell to see how the model performs on your question

#### Key Points:
- ONNX Runtime GenAI provides optimized inference for your model
- The model should run efficiently on CPU, without requiring a GPU
- The adapter contains the knowledge learned from the teacher model
- The model should respond with the correct multiple-choice answer most of the time


### Step 4: Register to Azure ML (5 min)

**Notebook:** `04.AzureML_RegisterToAzureML.ipynb`

**Purpose:** Register your optimized model to Azure ML for version tracking, sharing, and future deployment.

#### Instructions:

1. **Open the notebook** from the file explorer

2. **Install required packages**
   - Run the package installation cell if needed:
   ```python
   pip install azure-ai-ml
   pip install azure-identity
   pip install python-dotenv
   ```

3. **Import libraries and load environment**
   - Run the import cells:
   ```python
   import os
   from dotenv import load_dotenv
   from azure.ai.ml import MLClient
   from azure.ai.ml.entities import Model
   from azure.ai.ml.constants import AssetTypes
   from azure.identity import DefaultAzureCredential
   ```
   - Run the environment loading cell:
   ```python
   load_dotenv()
   
   subscription_id = os.getenv('AZUREML_SUBSCRIPTION_ID')
   resource_group = os.getenv('AZUREML_RESOURCE_GROUP')
   workspace = os.getenv('AZUREML_WS_NAME')
   
   print(f"Subscription ID: {subscription_id}")
   print(f"Resource Group: {resource_group}")
   print(f"Workspace Name: {workspace}")
   ```
   - Verify your Azure ML workspace information is displayed correctly

4. **Create ML client and connect to Azure ML**
   - Run the ML client creation cell:
   ```python
   ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
   print("Successfully connected to Azure ML workspace")
   ```

5. **Register your model**
   - Run the model registration cell:
   ```python
   file_model = Model(
       path="models/phi-4-mini/onnx",
       type=AssetTypes.CUSTOM_MODEL,
       name="fine-tuning-phi-4-mini-onnx-int4-cpu",
       description="Fine tuning by MSOlive",
   )
   
   registered_model = ml_client.models.create_or_update(file_model)
   print(f"Model registered with name: {registered_model.name}, version: {registered_model.version}")
   ```
   - Wait for confirmation that the model was registered successfully

6. **Verify registration**
   - Run the cell to list registered models:
   ```python
   models = list(ml_client.models.list())
   for model in models:
       print(f"Model: {model.name}, Version: {model.version}")
   ```
   - Verify your model appears in the list

#### Key Outputs:
- Your model registered in Azure ML with a unique name and version
- This registration makes your model discoverable and shareable with others
- You'll use this registered model for download in the next notebook


### Step 5: Download the Model (5 min)

**Notebook:** `05.Local_Download.ipynb`

**Purpose:** Download the registered model from Azure ML to your local machine for local deployment and inference.

#### Instructions:

1. **Important:** This notebook should be run on your local machine, not in Azure ML Studio
   - Download the notebook from the file explorer in Azure ML to your local computer
   - Open it in a local Jupyter environment (VS Code, JupyterLab, etc.)

2. **Install required packages**
   - Run the package installation cell:
   ```python
   import sys
   import subprocess
   
   def install_package(package_name):
       try:
           __import__(package_name)
           print(f"✓ {package_name} is already installed")
       except ImportError:
           print(f"Installing {package_name}...")
           subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
           print(f"✓ {package_name} installed successfully")
   
   install_package('python-dotenv')
   install_package('azure-identity')
   install_package('azure-ai-ml')
   ```

3. **Create local.env file**
   - Create a `local.env` file in the same directory as the notebook
   - Add your Azure ML credentials to the file:
   ```
   AZUREML_SUBSCRIPTION_ID=your-subscription-id
   AZUREML_RESOURCE_GROUP=your-resource-group
   AZUREML_WS_NAME=your-workspace-name
   ```

4. **Load environment and create ML client**
   - Run the environment loading cells
   - Run the ML client creation cell:
   ```python
   ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
   print("Successfully connected to Azure ML workspace")
   ```
   - Verify successful connection to Azure ML

5. **List available models**
   - Run the model listing cell:
   ```python
   models = list(ml_client.models.list())
   for model in models:
       print(f" - {model.name} (version: {model.version})")
   ```
   - Confirm your registered model is in the list

6. **Download your model**
   - Run the model download cell:
   ```python
   model_name = "fine-tuning-phi-4-mini-onnx-int4-cpu"
   model_version = 1
   
   print(f"Starting download of model: {model_name} (version {model_version})")
   
   download_path = ml_client.models.download(name=model_name, version=model_version)
   print(f"Model downloaded to: {download_path}")
   ```
   - Wait for the download to complete (may take a few minutes)

7. **Calculate model statistics**
   - Run the cells to calculate and display model size and file count
   - Note the total size, which should be much smaller than the original model

#### Key Outputs:
- A downloaded model on your local machine
- Information about the model size and files
- The model location for use in local inference


### Step 6: Local Inference (10 min)

**Notebook:** `06.Local_Inference.ipynb`

**Purpose:** Run the optimized model on your local machine to demonstrate its ability to answer questions without cloud resources.

#### Instructions:

1. **Run the notebook locally**
   - Open the notebook on your local machine in a Jupyter environment
   - This notebook should be run on the same machine where you downloaded the model

2. **Install ONNX Runtime GenAI**
   - Run the package installation cell:
   ```python
   def install_package(package_name):
       try:
           __import__(package_name)
           print(f"✓ {package_name} is already installed")
       except ImportError:
           print(f"Installing {package_name}...")
           subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
           print(f"✓ {package_name} installed successfully")

   install_package('onnxruntime-genai')
   ```

3. **Set the model path**
   - Update the model path to where your model was downloaded:
   ```python
   model_path = "./fine-tuning-phi-4-mini-onnx-int4-cpu/1/model"
   
   # Verify the model files exist
   if os.path.exists(model_path):
       print(f"Model found at path: {model_path}")
       print("Model files:")
       for file in os.listdir(model_path):
           print(f" - {file}")
   ```
   - Verify the model files are found

4. **Load the model and adapter**
   - Run the model loading cells:
   ```python
   model = og.Model(model_path)
   print("✓ Model loaded successfully!")
   
   adapters = og.Adapters(model)
   adapter_path = os.path.join(model_path, "adapter_weights.onnx_adapter")
   adapters.load(adapter_path, "qa_choice")
   ```
   - Confirm the model and adapter loaded successfully

5. **Set up the tokenizer**
   - Run the tokenizer setup cell:
   ```python
   tokenizer = og.Tokenizer(model)
   tokenizer_stream = tokenizer.create_stream()
   
   search_options = {}
   search_options['max_length'] = 102
   search_options['past_present_share_buffer'] = False
   search_options['repeat_penalty'] = 1.1
   search_options['temperature'] = 0.7
   ```

6. **Run inference on test questions**
   - Run the test question cells:
   ```python
   # Define some test questions
   test_questions = [
       {
           "question": "What is the capital of France?",
           "choices": {
               "A": "Berlin",
               "B": "London",
               "C": "Paris",
               "D": "Madrid",
               "E": "Rome"
           }
       },
       # ... other questions ...
   ]

   # Generate responses for each question
   for i, test_q in enumerate(test_questions):
       print(f"\n--- Question {i+1} ---")
       response = generate_response(test_q["question"], test_q["choices"])
       print(f"Final answer: {response}")
   ```
   - Review the model's answers to see if they're correct

7. **Try your own questions**
   - Run the custom question cell:
   ```python
   ask_question(
       "What is the main purpose of knowledge distillation in machine learning?",
       {
           "A": "To make models physically smaller in file size",
           "B": "To transfer knowledge from larger models to smaller ones",
           "C": "To increase the number of parameters in a model",
           "D": "To make training data more compact",
           "E": "To replace human knowledge with AI"
       }
   )
   ```
   - Or modify it to ask your own question

#### Key Achievements:
- Running an ML model entirely on your local machine
- Achieving fast inference without cloud resources
- Using significantly less memory than the original model
- Getting accurate answers to multiple-choice questions

## What You've Learned

Congratulations! You've successfully completed the Model Distillation Workshop. Here's what you've accomplished:

1. **Generated training data** using a large teacher model (DeepSeek-V3)
2. **Fine-tuned a smaller student model** (Phi-4-mini) using LoRA
3. **Optimized the model** with ONNX conversion and int4 quantization 
4. **Tested the model** using ONNX Runtime GenAI
5. **Registered the model** to Azure ML
6. **Downloaded and run the model locally**

You've learned:
- How knowledge distillation transfers intelligence from large to small models
- How to use Microsoft Olive for fine-tuning and optimization
- How to use ONNX Runtime GenAI for efficient inference
- How to deploy models locally for edge computing scenarios

## Next Steps

Here are some ways to build on what you've learned:

1. **Try different datasets**
   - Use different types of questions or tasks
   - Create your own custom dataset

2. **Explore different models**
   - Try different teacher models (GPT-4, Claude, etc.)
   - Try different student models (Phi-3, Llama, etc.)

3. **Optimize for different targets**
   - Try different quantization levels (int8, fp16)
   - Target different hardware (ARM, NVIDIA Jetson, etc.)

4. **Build applications**
   - Create a simple web UI for your model
   - Integrate it with other applications

5. **Learn more about:**
   - [Microsoft Olive](https://github.com/microsoft/Olive)
   - [ONNX Runtime](https://onnxruntime.ai/)
   - [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/)

Thank you for participating in this workshop! Your feedback is valuable for improving future sessions.
