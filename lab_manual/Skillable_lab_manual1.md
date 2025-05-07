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

# Knowledge Distillation with Azure ML

This notebook (`01.AzureML_Distillation.ipynb`) implements a knowledge distillation pipeline that leverages a teacher model from Azure AI to generate high-quality responses for multiple-choice questions from the commonsense QA dataset.

## Purpose

Knowledge distillation enables transferring knowledge from a large, powerful model (teacher) to a smaller, more efficient model (student). This notebook focuses on the first phase of this process:

1. Loading a dataset from Hugging Face
2. Preparing the data for inferencing
3. Using an Azure AI model as the teacher to generate answers
4. Analyzing the generated responses for quality and bias
5. Saving the resulting data for subsequent student model training

## Workflow Overview

1. **Environment Setup**: Installing required libraries and configuring authentication
2. **Dataset Handling**: Loading and preprocessing the commonsense QA dataset from Hugging Face
3. **Data Preparation**: Formatting questions as chat completions inputs
4. **Teacher Model Inference**: Generating answers using Azure AI Inference API
5. **Results Analysis**: Visualizing answer distributions and potential biases
6. **Data Export**: Saving processed Q&A pairs for student model training

## Technical Components

### Environment Setup
- Installation of essential Python packages:
  - `python-dotenv` for environment variable management
  - `datasets` for Hugging Face dataset access
  - `azure-ai-inference` for connecting to Azure AI services
- Loading environment variables for secure access to Azure AI services

### Dataset Handling
- Uses a custom `CQnAHuggingFaceInputDataset` class to load the "tau/commonsense_qa" dataset
- Implements customizable sample size selection for training, validation, and test data
- Handles dataset splits appropriately based on availability

### Data Preparation
- Creates JSONL format files with standardized message structure
- Implements a specific system prompt instructing the model to respond with A, B, C, D, or E answers
- Formats user prompts with question text and structured answer choices

### Teacher Model Inference
- Connects to Azure AI services using endpoint URL and API key stored in environment variables
- Uses `ChatCompletionsClient` from the Azure AI Inference SDK
- Implements a question processing function with proper error handling
- Displays progress with tqdm for visual feedback during batch processing

### Results Analysis
- Visualizes the distribution of generated answers
- Compares actual vs. expected answer distributions to identify biases
- Analyzes answer patterns by question type and length
- Creates heatmaps to identify potential biases in the teacher model's responses

### Data Export
- Saves processed question-answer pairs in JSONL format
- Extracts and formats only the essential information needed for training

## Code Highlights

### Dataset Loading
```python
train, val, _ = input_dataset.load_hf_dataset(
    dataset_name=dataset_name,
    train_sample_size=train_sample_size,
    val_sample_size=val_sample_size,
    train_split_name="train",
    val_split_name="validation",
)
```

### Azure AI Model Configuration
```python
endpoint = teacher_model_endpoint_url
model_name = teacher_model_name
key = teacher_model_api_key
client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
```

### Question Processing Function
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
            max_tokens=4000
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
```

## Benefits of This Approach

1. **Quality Data Generation**: Leverages a powerful teacher model to produce high-quality answers
2. **Focused Task Learning**: Structures the input with specific prompts for consistent output format
3. **Bias Detection**: Includes visualizations to identify and understand model biases
4. **Reproducibility**: Structured approach makes the process repeatable with different datasets
5. **Pipeline Integration**: Output format supports seamless integration with subsequent fine-tuning steps

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


## Next Steps

After generating the distillation dataset with this notebook, the data can be used in subsequent notebooks for:

1. Fine-tuning a smaller student model
2. Converting the model to an optimized format
3. Deploying the student model for inference
4. Evaluating the student model's performance against the teacher

This approach enables knowledge transfer from large, resource-intensive models to more efficient models suitable for deployment in resource-constrained environments while maintaining comparable task performance.

