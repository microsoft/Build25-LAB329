# Azure AI Foundry Setup for Lab329: H100 Compute & Deepseek Model

This guide provides step-by-step instructions for:
1. Setting up an NVIDIA A100 or H100 GPU compute node through Azure AI Foundry
2. Deploying the model using Models as a Service (MaaS) in supported regions

## Prerequisites

- An active Azure subscription
- Owner or Contributor access to your Azure subscription
- An existing Azure AI Foundry project or permission to create one

## Automated deployment using AZD/Bicep

We have provided an automated setup process using azd/bicep the setup configuration is located the `infra` folder 
you can update the machine sizes and specification by editing the following in the `main.bicep` file.

```
@description('VM size for the compute instance')
param computeVmSize string = 'Standard_ND96amsr_A100_v4' // High-performance GPU for model training using A100
param computeVmSize string = 'STANDARD_NC40ADS_H100_V5' // High-performance GPU for model training using H100 (Default used by the Bicep Deployment)

```

# Manual Setup Process 

## Project Setup and Supported Regions

### Supported Regions for Lab329

This lab requires specific Azure regions that support both A100 GPUs and MAI-DS-R1 model deployments:

- West US (westus)
- South Central US (southcentralus)
- East US (eastus)
- West US 3 (westus3)
- North Central US (northcentralus)
- East US 2 (eastus2)

### Step 1: Access Azure AI Foundry

1. Sign in to the [Azure Portal](https://portal.azure.com/)
2. Search for "AI Foundry" in the search bar
3. Select **Azure AI Foundry** from the results
   - If you don't see Azure AI Foundry, you may need to create an Azure AI Studio resource first

### Step 2: Create or Select an Azure AI Foundry Project

1. From the Azure AI Foundry dashboard, click **+ New project**
2. Fill in the required information:
   - **Project name**: Enter a descriptive name for your project (e.g., "Lab329-AI-Project")
   - **Subscription**: Select your Azure subscription
   - **Resource group**: Create new or select an existing resource group
   - **Region**: Select one of the supported regions listed above (important!)
   - **Storage account**: Create new or use an existing account
   - **Key Vault**: Create new or use an existing Key Vault
   - **Application Insights**: Create new or use an existing resource
3. Click **Create** to create your project
4. Wait for the project creation to complete

## Step 3: Configure Compute Resources

1. In your Azure AI Foundry project, navigate to the **Compute** section in the left sidebar
2. Click **+ New** to create a new compute resource
3. Select **Compute Cluster** as the compute type
4. Configure the following settings:
   - **Compute name**: Provide a unique name (e.g., "a100-compute")
   - **Location**: Should match your workspace region
   - **Virtual machine tier**: Select **GPU**
   - **Virtual machine type**: Choose **NCA100 v4-series** (This is the A100 series) or **NC40ADS_H100_V5** (This the H100 Series)
   - **Virtual machine size**: 
     - For single A100 GPU: Select **Standard_NC24ads_A100_v4** (1 A100 GPU) or **Standard_NC40ADS_H100_V5**(1 H100 GPU)
     - For multiple A100 GPUs machines/cluster: Select **Standard_NC48ads_A100_v4** (2 A100 GPUs) or higher
   - **Compute mode**: Select **Dedicated** for production workloads
   - **Minimum number of nodes**: Set to 0 to avoid idle costs
   - **Maximum number of nodes**: Set according to your needs and quota limits
   - **Idle time before scale down**: Set a reasonable timeout (e.g., 30 minutes)
   - **Advanced settings**:
     - Enable SSH access (optional but recommended for troubleshooting)
     - Configure network settings if needed

5. Click **Create** to provision your A100 compute resource
6. Wait for the compute to be created and reach "Running" state

## Step 4: Verify A100/H100 Compute Availability

1. Once the compute resource is created, it will appear in your compute list
2. The status should change to **Creating** and then to **Running**
3. You can verify the A100 GPU specification by clicking on the compute resource name

## Step 5: Use Your A100/H100 Compute with Azure ML Studio

### Option 1: Through Notebooks

1. Navigate to the **Notebooks** section in your Azure AI Foundry project
2. Create a new notebook or open an existing one
3. In the notebook, click on the compute selector in the top right
4. Select your A100/H100 compute from the dropdown menu
5. Run a simple test to verify GPU availability:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    # Should show "NVIDIA A100 80GB PCIe" or similar
```

## Important Considerations for A100 GPU Usage

### Cost Management

A100 GPUs are premium resources with significant hourly costs:

1. **Set minimum nodes to 0**: This ensures the compute scales down when not in use
2. **Monitor usage**: Regularly check your compute usage in the Azure Portal
3. **Set budget alerts**: Configure Azure budget alerts to avoid unexpected charges
4. **Shut down when not needed**: If you won't use the compute for days, consider deleting it

### Performance Optimization

1. **Batch size**: A100 GPUs have large memory (40GB or 80GB), optimize batch sizes accordingly
2. **Mixed precision training**: Use half-precision (FP16) or bfloat16 for better performance
3. **Distributed training**: For large models, configure distributed training across multiple A100s

### Quota Limitations

1. **Check your quota**: Verify your subscription has sufficient quota for A100/H100 GPUs
2. **Request increases**: If needed, request quota increases through the Azure Portal
3. **Regional availability**: A100 GPUs may not be available in all regions

## Troubleshooting

### Common Issues and Solutions

1. **"No quota available"**:
   - Request a quota increase through Azure Portal
   - Try a different region where A100s are available

2. **Compute creation fails**:
   - Verify your subscription has GPU quota
   - Check that you've selected a region where A100s are available
   - Ensure you have proper permissions on the resource group

3. **Long provisioning times**:
   - A100 GPUs are high-demand resources; provisioning may take 10-15 minutes
   - Be patient during the initial setup

4. **CUDA not available in notebooks**:
   - Verify you selected the correct compute in the notebook
   - Check if the necessary CUDA libraries are installed in your environment
   - Try using a curated environment with GPU support

## Azure Developer CLI (azd) Infrastructure Deployment

This lab includes Azure Bicep templates and Azure Developer CLI (azd) configuration files to automate the deployment of all required infrastructure components. This approach provides a consistent, repeatable deployment experience.

### Prerequisites for Infrastructure Deployment

1. **Azure CLI** - Install the latest version
   ```powershell
   winget install -e --id Microsoft.AzureCLI
   ```

2. **Azure Developer CLI (azd)** - Install the latest version
   ```powershell
   winget install -e --id Microsoft.Azd
   ```

3. **Azure Subscription** - An active Azure subscription with permissions to create resources

4. **Logged in to Azure** - Ensure you're logged in
   ```powershell
   az login
   ```

### Step-by-Step Deployment Using Azure Developer CLI

1. **Clone the Repository** (if not already done)
   ```powershell
   git clone https://github.com/yourusername/Build25-LAB329.git
   cd Build25-LAB329
   ```

2. **Initialize the Azure Developer CLI Environment**
   ```powershell
   azd init
   ```
   This will prompt you for an environment name, which will be used as a prefix for all Azure resources created.

3. **Deploy the Infrastructure**
   ```powershell
   azd up
   ```
   During deployment, you'll be prompted to provide:   - Azure subscription to use
   - Azure region for deployment (choose from supported regions)
   - DeepSeek-V3 model location (must support Azure AI services)
   - Compute VM size (defaults to Standard_ND96amsr_A100_v4 for GPU acceleration)

4. **Generate Environment Variables for Notebooks**
   ```powershell
   azd env get-values > Lab329\Notebook\local.env
   ```
   This command captures all output variables from the deployment and stores them in a file that the notebooks can use.

### What Gets Deployed

The azd deployment creates the following Azure resources:

1. **Resource Group** - Named `rg-<environmentName>`

2. **Azure Key Vault** - For secure storage of credentials and secrets

3. **Azure Storage Account** - Used by Azure ML for datasets, models, and outputs

4. **Azure AI Services** - DeepSeek-V3 model deployment (teacher model)

5. **Azure ML Hub Workspace** - For centralized machine learning operations

6. **Azure ML Project Workspace** - For running the knowledge distillation pipeline

7. **Compute Instance** - H100 compute For executing the notebooks and training processes

### Verification and Troubleshooting

1. **Verify Resource Deployment**:
   ```powershell
   azd env get-resources
   ```
   This shows all resources that have been deployed in your environment.

2. **Check Deployment Logs**:
   ```powershell
   azd env get-logs
   ```
   If deployment fails, this command helps identify the cause.

3. **Common Deployment Issues**:
   - **Quota limits**: Ensure your subscription has quota for the requested resources
   - **Region availability**: Verify the selected region supports all required services
   - **Permission issues**: Ensure your account has Contributor access to the subscription
   - **Name conflicts**: If resource names conflict, try a different environment name

### Accessing Deployed Resources

1. **Azure Portal**:
   - Navigate to the [Azure Portal](https://portal.azure.com/)
   - Search for and select the resource group named `rg-<environmentName>`
   - Explore the deployed resources within this group

2. **Azure ML Studio**:
   - In the Azure Portal, find your Azure ML workspace
   - Click "Launch Studio" to open Azure ML Studio
   - Access notebooks, compute, and other ML resources

### Clean Up

When you've completed the lab and want to remove all deployed resources:

```powershell
azd down
```

This command will delete all Azure resources created during deployment, preventing further charges.

## Resources

- [Azure AI Foundry Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- [NVIDIA A100 GPU Documentation](https://www.nvidia.com/en-us/data-center/a100/)
- [Azure NC A100 v4-series](https://learn.microsoft.com/en-us/azure/virtual-machines/nc-a100-v4-series)
- [Distributed Training Best Practices](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-distributed-gpu)

## Deploying Models Using Azure AI Foundry Models as a Service (MaaS)

Azure AI Foundry provides a serverless deployment option called Models as a Service (MaaS) that allows you to quickly deploy models without provisioning infrastructure. This section guides you through deploying models using MaaS in supported regions: westus, southcentralus, eastus, westus3, northcentralus, and eastus2.

### Prerequisites for MaaS Deployment

- An active Azure subscription
- Access to Azure AI Foundry
- Your model registered in Azure ML registry or a model ID from the Azure AI model catalog

### Step 1: Access the Model Deployment Page

1. Sign in to [Azure AI Studio](https://ai.azure.com/)
2. Navigate to your Azure AI Foundry project
3. Select **Models** from the left navigation menu
4. Find your model in the list or search for it
   - For Azure AI catalog models Deepseek-v3, you can use the model ID: `azureml://registries/azureml/models/deepseek-v3/versions/1`

### Step 2: Deploy the Model as a Service

1. Select your model from the list
2. Click on the **Deploy** button
3. Select **Real-time endpoint** as the deployment type
4. In the configuration page, select **Serverless** as the compute type
   - This is the Models as a Service option
5. Configure your deployment:
   - **Name**: Provide a unique name for your endpoint
   - **Region**: Select one of the supported regions: westus, southcentralus, eastus, westus3, northcentralus, or eastus2
   - **Authentication type**: Choose "Key-based authentication" or "Azure AD authentication" based on your security requirements
   - **Scaling settings**: Configure the scaling options based on your expected traffic
   - **Advanced settings**: Configure any advanced options as needed

6. Click **Create** to deploy the model

### Step 3: Test the Deployed Endpoint

1. Once deployment is complete, navigate to the **Endpoints** section in your Azure AI Foundry project
2. Select your newly created endpoint
3. Go to the **Test** tab
4. Enter a sample input or use the provided examples
5. Click **Test** to verify that your endpoint is working properly

### Step 4: Get Deployment Details for Integration

1. Go to the **Consume** tab of your endpoint
2. Here you'll find:
   - **REST endpoint URL**: The URL to send API requests to
   - **API keys**: Authentication keys for your endpoint
   - **Sample code**: Code snippets for Python, C#, and other languages to integrate with your endpoint

### Step 5: Monitor Your MaaS Deployment

1. Go to the **Metrics** tab of your endpoint
2. Here you can monitor:
   - **Request count**: Number of requests over time
   - **Latency**: Response time for requests
   - **Token usage**: Number of tokens consumed
   - **Error rate**: Percentage of failed requests

### Cost Considerations for MaaS

Unlike traditional compute resources that charge by uptime, MaaS follows a pay-as-you-go model:

1. **Token-based pricing**: You pay only for the tokens consumed (input + output)
2. **No idle costs**: No charges when your endpoint isn't processing requests
3. **Regional pricing**: Costs may vary by region, check the [Azure pricing page](https://azure.microsoft.com/en-us/pricing/details/machine-learning/) for details

### Supported Model Types for MaaS

Not all models can be deployed using MaaS. Currently, supported models include:

1. Foundation models from the Azure AI model catalog
2. Custom models that meet specific requirements:
   - Compatible ML frameworks (PyTorch, ONNX, etc.)
   - Within size limitations (check documentation for current limits)
   - Properly packaged with scoring script and environment definition

### Troubleshooting MaaS Deployments

1. **Deployment fails**:
   - Verify you've selected a supported region
   - Check if the model is compatible with MaaS
   - Review any error messages in the deployment logs

2. **High latency**:
   - MaaS endpoints may have higher cold-start times than dedicated compute
   - Consider using dedicated compute for latency-sensitive applications

3. **Authentication issues**:
   - Verify you're using the correct authentication method (key vs. Azure AD)
   - Check that your API keys haven't expired
   - Ensure proper CORS settings if calling from web applications

### Connecting Deepseek-v3 Model via MaaS

For the specific Deepseek model mentioned in the lab materials:

1. Use the model ID: `azureml://registries/azureml/models/deepseek-v3/versions/1`
2. Deploy to one of the supported regions: westus, southcentralus, eastus, westus3, northcentralus, or eastus2
3. Use the following Python code to connect to your deployed endpoint:

```python
import json
import requests

# Replace with your endpoint details
endpoint_url = "YOUR_ENDPOINT_URL"
api_key = "YOUR_API_KEY"

# Prepare headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}" # Use this for Azure AD auth
    # "api-key": api_key # Use this for key-based auth
}

# Prepare data
data = {
    "messages": [
        {"role": "system", "content": "You are an AI assistant that helps with reasoning tasks."},
        {"role": "user", "content": "Solve this step by step: If a^2 + b^2 = 25 and ab = 12, find a + b."}
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 800
}

# Make the request
response = requests.post(endpoint_url, headers=headers, data=json.dumps(data))

# Process the response
if response.status_code == 200:
    response_data = response.json()
    print(response_data["choices"][0]["message"]["content"])
else:
    print(f"Error: {response.status_code}, {response.text}")
```

## Setting Up IAM Permissions for Azure ML Compute

Proper permissions are essential for users to work with Azure ML compute resources. This section guides you through setting up Identity and Access Management (IAM) permissions for your Azure ML resources.

### Step 1: Configure Resource Group-Level Permissions

1. Sign in to the [Azure Portal](https://portal.azure.com/)
2. Navigate to the Resource Group containing your Azure AI Foundry project
3. Select **Access control (IAM)** from the left sidebar
4. Click on **+ Add** and then **Add role assignment**
5. Configure the following settings:
   - **Role**: Select one of the following based on user needs:
     - **Contributor**: Grants full access to manage all resources (recommended for lab admins)
     - **AzureML Data Scientist**: Can perform most Azure ML operations, but can't create compute resources
     - **AzureML Compute Operator**: Can only use existing compute resources
   - **Assign access to**: Select "User, group, or service principal"
   - **Select**: Search for and select the user account(s) that need access
6. Click **Review + assign** to save the role assignment

### Step 2: Configure Azure ML Workspace-Level Permissions

For more granular control, you can assign roles at the workspace level:

1. Navigate to your Azure ML workspace resource in the Azure Portal
2. Select **Access control (IAM)** from the left sidebar
3. Click on **+ Add** and then **Add role assignment**
4. Assign one of the following roles based on user needs:
   - **Owner**: Full control of the workspace, including user management
   - **Contributor**: Can create and manage all resources in the workspace
   - **AzureML Data Scientist**: Can submit runs and create experiments, but can't create compute
   - **AzureML Compute Operator**: Can only use existing compute resources
5. Select the user(s) and click **Review + assign**

### Step 3: Configure Compute-Specific Permissions

To provide access to specific compute resources only:

1. Navigate to your Azure AI Foundry project
2. Select **Compute** from the left sidebar
3. Click on the specific compute cluster you want to share
4. Select the **Access control (IAM)** tab
5. Click **+ Add** and then **Add role assignment**
6. Assign the **AzureML Compute Operator** role to specific users
7. Click **Review + assign**

### Step 4: Verify Permissions

To verify that permissions have been correctly assigned:

1. Ask users to sign into the [Azure AI Studio](https://ai.azure.com/)
2. Navigate to the project and try to access compute resources
3. Verify they can perform actions appropriate to their assigned role:
   - Viewing compute resources
   - Submitting jobs to compute clusters
   - Creating notebooks and connecting to compute instances

### Common Permission Issues and Troubleshooting

1. **"No permission to create compute" error**:
   - Ensure the user has at least Contributor role on the resource group
   - Check if there are Azure Policy restrictions in place

2. **"Cannot access storage account" error**:
   - Ensure the user has the Storage Blob Data Contributor role on the associated storage account

3. **"Failed to start notebook" error**:
   - Check if the user has permissions to the underlying compute resource
   - Verify the user has AzureML Data Scientist role at minimum

4. **Permission changes not taking effect immediately**:
   - Role assignments can take a few minutes to propagate
   - Ask users to sign out and sign back in to refresh their token

### Best Practices for IAM in Azure ML

1. **Use Azure AD groups**: Assign permissions to Azure AD groups rather than individual users for easier management
2. **Follow least privilege principle**: Assign the minimum permissions required for users to perform their tasks
3. **Regular access review**: Periodically review and update access permissions to maintain security
4. **Use custom roles**: For advanced scenarios, create custom IAM roles with precisely defined permissions

### Required Permissions for Common Tasks

| Task | Required Role |
|------|--------------|
| Create compute resources | Contributor or Owner at resource group level |
| Use existing compute | AzureML Compute Operator |
| Submit training jobs | AzureML Data Scientist |
| Deploy models | AzureML Data Scientist + additional deployment permissions |
| View experiment results | Reader |
| Manage security settings | Owner |
