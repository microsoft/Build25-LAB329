# Deploying Azure Infrastructure for Model Distillation Workshop

This guide provides instructions for deploying the Azure infrastructure required for the Model Distillation Workshop using Bicep templates.

## Infrastructure Overview

The deployment includes:
- Azure AI Foundry (Hub & Project)
- Azure AI Foundry for DeepSeek V3 model
- Azure ML Compute resources for model training and inference
- Storage accounts and Key Vaults for secure data management
- Necessary role assignments for service access

## Deployment Options

### Option 1: Using Azure Developer CLI (azd) - Recommended

The Azure Developer CLI (azd) is the simplest way to deploy your infrastructure as it handles parameter substitution and resource naming automatically based on your azure.yaml file.

#### Prerequisites:
- Install Azure Developer CLI from https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd
- Log in to Azure using `azd auth login`

#### Deployment Steps:

1. Navigate to the root directory of your project (where azure.yaml is located)
   
   **PowerShell:**
   ```powershell
   cd path\to\Build25-LAB329
   ```
   
   **Bash:**
   ```bash
   cd path/to/Build25-LAB329
   ```

2. Initialize your environment (first time only)
   
   **PowerShell/Bash:**
   ```bash
   azd init
   ```

3. Deploy the infrastructure
   
   **PowerShell/Bash:**
   ```bash
   azd provision
   ```

During the `azd provision` process, you will be prompted to:
- Select a subscription
- Provide a unique environment name
- Choose a location for your resources
- (Optionally) confirm the deployment

### Option 2: Using Azure CLI

If you prefer to use Azure CLI directly:

1. Login to Azure
   
   **PowerShell/Bash:**
   ```bash
   az login
   ```

2. Set your subscription
   
   **PowerShell/Bash:**
   ```bash
   az account set --subscription "YOUR_SUBSCRIPTION_ID"
   ```

3. Create a resource group (if it doesn't exist)
   
   **PowerShell:**
   ```powershell
   $resourceGroupName = "rg-build25-lab329"
   $location = "eastus"  # Choose an appropriate region
   az group create --name $resourceGroupName --location $location
   ```
   
   **Bash:**
   ```bash
   resourceGroupName="rg-build25-lab329"
   location="eastus"  # Choose an appropriate region
   az group create --name $resourceGroupName --location $location
   ```

4. Deploy the Bicep template
   
   **PowerShell:**
   ```powershell
   az deployment sub create `
     --name "build25-lab329-deployment" `
     --location $location `
     --template-file "infra\main.bicep" `
     --parameters environmentName="build25lab329" `
                 location=$location `
                 deepSeekV31Location="eastus" `
                 principalId=$(az ad signed-in-user show --query id -o tsv)
   ```
   
   **Bash:**
   ```bash
   az deployment sub create \
     --name "build25-lab329-deployment" \
     --location $location \
     --template-file "infra/main.bicep" \
     --parameters environmentName="build25lab329" \
                 location=$location \
                 deepSeekV31Location="eastus" \
                 principalId=$(az ad signed-in-user show --query id -o tsv)
   ```

## Important Parameters

The Bicep deployment requires several parameters:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| environmentName | A name for your environment (used in resource naming) | "build25lab329" |
| location | Primary location for resources | "eastus", "westus2" |
| deepSeekV31Location | Location for the DeepSeek model (must support the model) | "eastus" |
| principalId | Object ID of the user/service principal deploying resources | *auto-generated* |
| computeVmSize | VM size for Azure ML compute | "STANDARD_NC40ADS_H100_V5" |

## After Deployment

After successful deployment, the outputs include:
- Azure ML Workspace information
- Teacher model endpoint and key
- Azure AI Project connection string

These outputs can be used to create a local.env file for the workshop notebooks:

```
TEACHER_MODEL_NAME=DeepSeek-V3
TEACHER_MODEL_ENDPOINT=<from-deployment-output>
TEACHER_MODEL_KEY=<from-deployment-output>
AZUREML_SUBSCRIPTION_ID=<from-deployment-output>
AZUREML_RESOURCE_GROUP=<from-deployment-output>
AZUREML_WS_NAME=<from-deployment-output>
```

### Accessing Deployment Outputs

#### With Azure Developer CLI:

**PowerShell:**
```powershell
# View all outputs
azd env get-values

# Get a specific output
azd env get-values | findstr TEACHER_MODEL_ENDPOINT
```

**Bash:**
```bash
# View all outputs
azd env get-values

# Get a specific output
azd env get-values | grep TEACHER_MODEL_ENDPOINT
```

#### With Azure CLI:

**PowerShell/Bash:**
```bash
# View all outputs
az deployment sub show --name "build25-lab329-deployment" --query "properties.outputs" -o json
```

## Troubleshooting

If deployment fails, check the following:
- Ensure you have sufficient permissions to create resources in your subscription
- Verify the selected regions support all required services
- Check that you have sufficient quota for the GPU compute resources
- Examine the deployment logs for specific error messages

## Resource Cleanup

When you're finished with the workshop, you can remove all resources to prevent further charges:

**PowerShell/Bash:**
```bash
# Using Azure Developer CLI
azd down
```

**PowerShell:**
```powershell
# Using Azure CLI
az group delete --name $resourceGroupName --yes --no-wait
```

**Bash:**
```bash
# Using Azure CLI
az group delete --name $resourceGroupName --yes --no-wait
```
