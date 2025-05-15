
# Step 5: Download the Model (5 min)

**Notebook:** `05.Local_Download.ipynb`

**Purpose:** Download the registered model from Azure ML to your Skillable VM for local deployment and inference. You will be running this notebook from the Skillable VM. Please open the folder `C:\Users\LabUser\Desktop\lab\Build25-LAB329\Lab329\Notebook` which is located on the desktop of your VM in a folder called `lab`.

# Local Model Download

This notebook (`05.Local_Download.ipynb`) focuses on downloading the fine-tuned and optimized model from Azure Machine Learning to a local environment for deployment or inference.

## Purpose

After fine-tuning and optimizing our model in the cloud, this notebook enables you to:
1. Download the ONNX-formatted model to your local environment
2. Access model files for deployment on edge devices or integration with local applications
3. Use the model in scenarios that require offline inference capabilities

## Requirements

- Python 3.10+ installed on your local machine
- Azure CLI installed and configured (`az login`)
- Access permissions to the Azure ML workspace containing the model

## Workflow

1. **Environment Setup**
   - Install required Python packages (`azure-ai-ml`, `azure-identity`, `python-dotenv`, etc.)
   - Import necessary libraries for Azure ML operations
   - Configure environment variables for Azure authentication
2. **Azure Authentication**
   - Connect to Azure using the Azure CLI authentication
   - Load credentials from a local environment file (`local.env`)
   - Establish connection to the Azure ML workspace
3. **Model Discovery**
   - List available models in the Azure ML workspace
   - Display model names and versions for selection
   - Validate access permissions and connection status
4. **Model Download**
   - Download the specified model version from Azure ML
   - Track download progress with text-based indicators
   - Handle download errors with informative messages
   - Support both synchronous and asynchronous download methods
5. **Download Verification**
   - Calculate model size and file count statistics
   - Format and display model details (size in appropriate units)
   - Verify download location and model accessibility
   - Handle edge cases where download path isn't explicitly returned

## Key Features

### Robust Error Handling
- Comprehensive error detection and recovery mechanisms
- User-friendly error messages with specific troubleshooting guidance
- Fallback mechanisms for different Azure ML SDK versions

### Environment Variable Management
- Flexible handling of configuration through `local.env` files
- Automatic creation of template files if none exist
- Validation of required environment variables

### Download Resilience
- Multiple download strategies (sync and async)
- Path verification and discovery if standard location fails
- Automatic conversion of size measurements to appropriate units

### Progress Monitoring
- Text-based progress indicators during download
- Download duration tracking and reporting
- Detailed model statistics after completion (size, file count)

## Technical Implementation

### Authentication
```python
# Create ML client with the credentials
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
```

### Model Discovery
```python
# List available models in the workspace
models = list(ml_client.models.list())
for model in models:
    print(f" - {model.name} (version: {model.version})")
```

### Download Process
```python
# Download the model
download_path = ml_client.models.download(name=model_name, version=model_version)
```

### Model Statistics
```python
# Calculate model statistics
total_size = 0
file_count = 0
for path, dirs, files in os.walk(download_path):
    for f in files:
        fp = os.path.join(path, f)
        total_size += os.path.getsize(fp)
        file_count += 1
```

## Troubleshooting Guide

The notebook includes specific troubleshooting guidance for common issues:

- **Authentication Issues**: Instructions for re-authenticating using `az login --use-device-code`
- **Missing Environment Variables**: Guidance on setting up the `local.env` file
- **Model Not Found**: Steps to verify model existence and permissions
- **Download Failures**: Alternative approaches and fallback mechanisms
- **Path Resolution Issues**: Techniques to locate downloaded model when path is not returned

## Next Steps

After downloading the model:

1. Use the model for local inference with ONNX Runtime or other compatible frameworks
2. Deploy the model to edge devices or embedded systems
3. Integrate the model with custom applications or services
4. Compare performance metrics between cloud deployment and local inference

This notebook completes the knowledge distillation pipeline by bringing the optimized model back to the local environment, where it can be deployed and used in various scenarios outside of Azure ML.