# Microsoft AI Foundry Hub Project Setup Guide

This guide provides step-by-step instructions for setting up a [Microsoft Azure AI Foundry](https://ai.azure.com) Hub Project, deploying a GPT-4o model, and configuring an A100 compute node. The guide is organized into logical sections to facilitate the setup process.

## Table of Contents
1. [Introduction to Microsoft AI Foundry](#introduction-to-microsoft-ai-foundry)
2. [Prerequisites](#prerequisites)
3. [Creating an AI Foundry Hub Project](#creating-an-ai-foundry-hub-project)
4. [Setting Up Azure Resources](#setting-up-azure-resources)
5. [Deploying GPT-4o Model](#deploying-gpt-4o-model)
6. [Configuring A100 Compute Node](#configuring-a100-compute-node)
7. [Testing the Deployment](#testing-the-deployment)
8. [Monitoring and Management](#monitoring-and-management)
9. [Automation with Bicep Templates](#automation-with-bicep-templates)
10. [Troubleshooting](#troubleshooting)

## Introduction to Microsoft AI Foundry

Microsoft AI Foundry is a platform designed to simplify the deployment and management of advanced AI models in Azure. It provides the infrastructure and tools needed to deploy and operate large language models like GPT-4o efficiently.

### Key Features
- Streamlined model deployment
- High-performance compute resources management
- Integrated monitoring and logging
- Cost optimization features
- Enterprise-grade security and compliance

## Prerequisites

Before you begin, ensure you have the following:

- An active Azure subscription with appropriate permissions
- Azure CLI installed and configured (version 2.40.0 or later)
- Bicep CLI installed (version 0.9.1 or later)
- Access to Microsoft AI Foundry service
- Quota for A100 GPUs in your target region
- Service principal with contributor access to your subscription
- Knowledge of your organization's networking requirements

### Required Permissions

- Contributor role on the subscription or resource group
- AI Foundry Administrator role
- Network Contributor (if deploying within an existing VNet)

## Creating an AI Foundry Hub Project

### Step 1: Access the AI Foundry Portal

1. Navigate to the [Azure Portal](https://portal.azure.com)
2. Search for "AI Foundry" in the search bar
3. Select "AI Foundry" from the results

### Step 2: Create a New Hub Project

1. Click "Create new project"
2. Enter a project name (must be globally unique)
3. Select your subscription and resource group (create a new one if needed)
4. Choose the region closest to your users (ensure A100 GPU availability)
5. Configure the networking settings:
   - Choose between Default, Private Endpoint, or VNET Integration
   - If using VNET, select or create a virtual network and subnet
6. Add tags for resource organization (optional)
7. Click "Review + create", then "Create"

### Step 3: Configure Hub Project Settings

1. Once deployed, navigate to your new [AI Foundry Hub Project](https://ai.azure.com)
2. Configure authentication settings
3. Set up user access by assigning appropriate roles
4. Configure logging and monitoring settings

## Setting Up Azure Resources

### Step 1: Create Resource Group (if not already done)

```bash
az group create --name aifoundry-rg --location eastus
```

### Step 2: Set Up Storage Account

```bash
az storage account create \
  --name aifoundrystorage \
  --resource-group aifoundry-rg \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2 \
  --enable-hierarchical-namespace true
```

### Step 3: Create Container Registry

```bash
az acr create \
  --name aifoundryacr \
  --resource-group aifoundry-rg \
  --location eastus \
  --sku Premium
```

## Deploying GPT-4o Model

### Step 1: Access Model Catalog

1. In your [Azure AI Foundry Hub Project](https://ai.azure.com), navigate to "Model Catalog"
2. Search for "GPT-4o" in the catalog
3. Select the GPT-4o model from the results

### Step 2: Configure Model Deployment

1. Click "Deploy model"
2. Configure deployment settings:
   - Deployment name: A unique name for your deployment

## Configuring A100 Compute Node

### Step 1: Set Up A100 Compute Cluster

1. In your [Azure AI Foundry Hub Project](https://ai.azure.com), navigate to "Compute"
2. Click "Create"
3. Select "A100" from the GPU options
4. Configure the cluster:
   - Cluster name: A unique name for your compute cluster
   - VM size: Select the appropriate size (ND A100 v4-series recommended)
   - Minimum nodes: Set to 0 for cost efficiency
   - Maximum nodes: Based on your workload and quota
   - Idle seconds before scale down: 300 (5 minutes) or adjust as needed
   - Advanced networking settings:
     - Virtual network and subnet configuration
     - Public IP settings
     - Security group rules

### Step 2: Optimize A100 Configuration

1. Configure CUDA settings for optimal performance
2. Set up node pool autoscaling based on workload
3. Configure memory allocation for large model operations
4. Set up monitoring and alerts for GPU utilization

### Step 3: Attach Compute to Model Deployment

1. Navigate to your GPT-4o deployment settings
2. Under "Compute configuration," select your A100 compute cluster
3. Save the configuration
4. Restart the deployment to apply changes

## Testing the Deployment

### Step 1: Basic Connectivity Testing

Test basic connectivity to your deployment:

```bash
curl -X POST https://{your-deployment-endpoint}/completions \
  -H "Content-Type: application/json" \
  -H "api-key: {your-api-key}" \
  -d '{
    "prompt": "Hello, world!",
    "max_tokens": 50
  }'
```

### Step 2: Performance Testing

Run performance tests to ensure your A100 compute node is properly configured:

1. Test response times under various loads
2. Monitor GPU utilization during inference
3. Test concurrent requests handling
4. Measure token processing speed

### Step 3: Integration Testing

Test integration with your applications:

1. Use the provided SDK to connect to your deployment
2. Test authentication and authorization flows
3. Verify error handling and retry logic

## Monitoring and Management

### Setting Up Monitoring

1. Navigate to "Monitoring" section in your AI Foundry Hub Project
2. Configure the following dashboards:
   - Model performance metrics
   - Compute utilization
   - Request volume and latency
   - Error rates and types

### Cost Management

1. Set up budget alerts in the Azure portal
2. Configure auto-scaling to optimize costs
3. Schedule regular cost reviews
4. Consider reserved instances for consistent workloads

### Security and Compliance

1. Review access controls regularly
2. Enable Private Link for secure connectivity
3. Configure data encryption settings
4. Set up audit logging for compliance requirements

## Automation with Bicep Templates

See the accompanying [ai_foundry_deployment.bicep](../infrastructure/ai_foundry_deployment.bicep) file for automated deployment.

To deploy using the Bicep template:

```bash
az deployment group create \
  --resource-group aifoundry-rg \
  --template-file ai_foundry_deployment.bicep \
  --parameters projectName=yourprojectname region=eastus
```

## Troubleshooting

### Common Issues

1. **Quota Limits**: Ensure you have sufficient quota for A100 GPUs in your target region.
   - Solution: Request a quota increase through the Azure portal.

2. **Networking Issues**: VNet integration problems.
   - Solution: Verify subnet has appropriate permissions and NSG rules.

3. **Model Deployment Failures**:
   - Check resource allocation
   - Verify model compatibility
   - Review deployment logs for specific error messages

4. **Performance Issues**:
   - Review compute configuration
   - Check for resource contention
   - Analyze inference optimization settings

### Getting Support

If you encounter issues not covered in this guide:

1. Check the [Microsoft AI Foundry documentation](https://learn.microsoft.com/azure/ai-foundry/)
2. Open a support ticket through the Azure portal
3. Post on Microsoft Q&A with tag "ai-foundry"
4. Check for service health notifications in the Azure portal

---

This guide is current as of April 2025. Microsoft AI Foundry services and features may change over time. Always refer to the official documentation for the most up-to-date information.
