# Microsoft Azure AI Foundry Deployment Guide

This guide provides step-by-step instructions for deploying the Microsoft Azure AI Foundry infrastructure using the Azure Bicep template (`ai_foundry_deployment.bicep`).

## Prerequisites

- **Azure CLI**: Install the [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- **Bicep CLI**: Install the [Bicep CLI](https://docs.microsoft.com/en-us/azure/azure-resource-manager/bicep/install)
- **Azure Subscription**: Active Azure subscription with sufficient permissions
- **Resource Quota**: Ensure you have quota for H100 GPUs in your target region (contact Azure support if needed)

## Resources Created

The deployment creates the following Azure resources:

- **Azure Machine Learning Workspace**: Microsoft Azure AI Foundry Hub project environment
- **Storage Account**: For storing models and training data (with HNS enabled)
- **Azure Container Registry**: For storing Docker images
- **Key Vault**: For secure storage of secrets
- **Log Analytics Workspace**: For monitoring and logs
- **GPU Compute Cluster**: H100 GPU cluster for model training and inference
- **Model Deployments**: 
  - Teacher model (Llama-4-Scout)
  - Student model (Phi-4)
- **Online Endpoints**: For model inferencing

## Deployment Steps

### 1. Sign in to Azure

```bash
az login
```

### 2. Set Default Subscription

```bash
az account set --subscription "<SUBSCRIPTION_ID>"
```

### 3. Create Resource Group

```bash
az group create --name <RESOURCE_GROUP_NAME> --location <LOCATION>
```

Example:
```bash
az group create --name ai-foundry-rg --location eastus
```

### 4. Deploy the Bicep Template

Create a parameters file named `parameters.json` with the following content:

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "projectName": {
      "value": "distillation"
    },
    "location": {
      "value": "eastus"
    },
    "environmentType": {
      "value": "dev"
    },
    "nodeCount": {
      "value": 1
    },
    "adminUsername": {
      "value": "aiadmin"
    },
    "adminPassword": {
      "value": "REPLACE_WITH_SECURE_PASSWORD"
    },
    "enablePrivateNetworking": {
      "value": false
    }
  }
}
```

> **Important**: Replace the `adminPassword` value with a secure password. It will be used for the compute node administration.

Run the deployment command:

```bash
az deployment group create \
  --resource-group <RESOURCE_GROUP_NAME> \
  --template-file ai_foundry_deployment.bicep \
  --parameters @parameters.json
```

Example:
```bash
az deployment group create \
  --resource-group ai-foundry-rg \
  --template-file ai_foundry_deployment.bicep \
  --parameters @parameters.json
```

### 5. Private Networking Configuration (Optional)

If you need to deploy with private networking:

1. First, create a Virtual Network and Subnet or use an existing one
2. Update the parameters.json file:

```json
{
  "enablePrivateNetworking": {
    "value": true
  },
  "vnetName": {
    "value": "my-vnet"
  },
  "subnetName": {
    "value": "default"
  },
  "vnetResourceGroupName": {
    "value": "network-rg"
  }
}
```

### 6. Verify Deployment

After deployment completes, check the outputs to get important resource names:

```bash
az deployment group show \
  --resource-group <RESOURCE_GROUP_NAME> \
  --name ai_foundry_deployment \
  --query "properties.outputs"
```

## Parameter Reference

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| projectName | Name for your Microsoft Azure AI Foundry project | (Required) |
| location | Azure region to deploy resources | Resource group location |
| environmentType | Environment type (dev, test, prod) | dev |
| gpuSku | GPU SKU for compute nodes | Standard_ND96isr_H100_v5 |
| nodeCount | Number of GPU nodes to deploy | 1 |
| adminUsername | Admin username for compute nodes | (Required) |
| adminPassword | Admin password for compute nodes | (Required) |
| enablePrivateNetworking | Enable private networking | false |
| vnetName | Virtual network name (for private networking) | '' |
| subnetName | Subnet name (for private networking) | '' |
| vnetResourceGroupName | Resource group for existing VNET | Current resource group |

## Post-Deployment Tasks

After deployment:

1. **Access Azure ML Studio**:
   - Navigate to the [Azure ML Studio](https://ml.azure.com)
   - Select your subscription and the newly created workspace

2. **Upload Training Data**:
   - Upload your training data to the `data` container in the storage account

3. **Monitor Deployments**:
   - Check the status of model deployments in the Azure ML workspace
   - Test the endpoints using the provided endpoint URLs

4. **Cost Management**:
   - Remember that H100 GPUs are expensive resources
   - Scale down or stop compute resources when not in use

## Troubleshooting

- **Deployment Failures**: Check the deployment logs in the Azure portal
- **Quota Issues**: Ensure you have sufficient quota for H100 GPUs
- **Private Networking**: If using private endpoints, ensure DNS resolution is properly configured

## Security Considerations

- Store sensitive parameter values like passwords in Azure Key Vault
- Consider using managed identities for authentication
- Review and apply appropriate RBAC permissions to resources

## Clean Up Resources

When finished, delete the resource group to avoid unnecessary charges:

```bash
az group delete --name <RESOURCE_GROUP_NAME> --yes --no-wait
```

## Additional Resources

- [Azure Bicep Documentation](https://docs.microsoft.com/en-us/azure/azure-resource-manager/bicep/)
- [Azure Machine Learning Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Azure ML Model Deployment](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints)
