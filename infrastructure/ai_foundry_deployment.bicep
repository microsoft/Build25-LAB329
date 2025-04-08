@description('Project name for the AI Foundry Hub')
param projectName string

@description('Azure region to deploy resources')
param location string = resourceGroup().location

@description('Environment type (dev, test, prod)')
@allowed([
  'dev'
  'test'
  'prod'
])
param environmentType string = 'dev'

@description('GPU SKU for compute nodes')
@allowed([
  'Standard_ND96asr_v4' // A100 GPU
  'Standard_ND96amsr_v4' // A100 GPU with more memory
])
param gpuSku string = 'Standard_ND96asr_v4'

@description('Number of GPU nodes to deploy')
@minValue(1)
@maxValue(10)
param nodeCount int = 1

@description('Administrator username for compute nodes')
param adminUsername string

@description('Administrator password for compute nodes')
@secure()
param adminPassword string

@description('Enable private networking')
param enablePrivateNetworking bool = false

@description('Virtual network name (required if enablePrivateNetworking is true)')
param vnetName string = ''

@description('Subnet name (required if enablePrivateNetworking is true)')
param subnetName string = ''

@description('Existing resource group for vnet (leave empty if creating new vnet)')
param vnetResourceGroupName string = resourceGroup().name

@description('Tags to apply to resources')
param tags object = {
  environment: environmentType
  project: projectName
}

// Variables
var uniqueSuffix = substring(uniqueString(resourceGroup().id), 0, 6)
var storageAccountName = 'aifoundry${uniqueSuffix}'
var acrName = 'aifoundryacr${uniqueSuffix}'
var aiFoundryProjectName = 'aifoundry-${projectName}-${environmentType}'
var logAnalyticsWorkspaceName = 'aifoundry-law-${uniqueSuffix}'
var keyVaultName = 'aifoundry-kv-${uniqueSuffix}'
var computeClusterName = 'a100-cluster'
var deploymentName = 'gpt4o-deployment'

// Log Analytics Workspace
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsWorkspaceName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// Storage Account
resource storageAccount 'Microsoft.Storage/storageAccounts@2022-09-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: environmentType == 'prod' ? 'Standard_GRS' : 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    isHnsEnabled: true
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

// Blob Services
resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2022-09-01' = {
  name: 'default'
  parent: storageAccount
  properties: {
    containerDeleteRetentionPolicy: {
      enabled: true
      days: 7
    }
  }
}

// Container for model data
resource modelContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-09-01' = {
  name: 'models'
  parent: blobServices
  properties: {
    publicAccess: 'None'
  }
}

// Container for training data
resource dataContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2022-09-01' = {
  name: 'data'
  parent: blobServices
  properties: {
    publicAccess: 'None'
  }
}

// Azure Container Registry
resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: acrName
  location: location
  tags: tags
  sku: {
    name: environmentType == 'prod' ? 'Premium' : 'Standard'
  }
  properties: {
    adminUserEnabled: true
    publicNetworkAccess: enablePrivateNetworking ? 'Disabled' : 'Enabled'
  }
}

// Key Vault
resource keyVault 'Microsoft.KeyVault/vaults@2022-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    accessPolicies: []
    enableRbacAuthorization: true
  }
}

// Virtual Network (if private networking is enabled)
resource vnet 'Microsoft.Network/virtualNetworks@2022-07-01' = if (enablePrivateNetworking && empty(vnetName)) {
  name: empty(vnetName) ? '${projectName}-vnet' : vnetName
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.0.0.0/16'
      ]
    }
    subnets: [
      {
        name: empty(subnetName) ? 'default' : subnetName
        properties: {
          addressPrefix: '10.0.0.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
          privateLinkServiceNetworkPolicies: 'Enabled'
        }
      }
    ]
  }
}

// AI Foundry Hub Project
resource aiFoundryProject 'Microsoft.MachineLearningServices/workspaces@2023-06-01-preview' = {
  name: aiFoundryProjectName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'AI Foundry Hub - ${projectName}'
    description: 'AI Foundry Hub Project for ${projectName}'
    storageAccount: storageAccount.id
    containerRegistry: acr.id
    keyVault: keyVault.id
    applicationInsights: logAnalyticsWorkspace.id
    publicNetworkAccess: enablePrivateNetworking ? 'Disabled' : 'Enabled'
    // Optional private endpoint configuration if enablePrivateNetworking is true
  }
}

// Compute Cluster for A100 GPUs
resource computeCluster 'Microsoft.MachineLearningServices/workspaces/computes@2023-06-01-preview' = {
  name: computeClusterName
  parent: aiFoundryProject
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: gpuSku
      vmPriority: 'Dedicated'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: nodeCount
        nodeIdleTimeBeforeScaleDown: 'PT5M'
      }
      remoteLoginPortPublicAccess: 'NotSpecified'
      adminUserName: adminUsername
      adminUserPassword: adminPassword
      subnet: enablePrivateNetworking ? {
        id: enablePrivateNetworking && empty(vnetName) 
          ? vnet.properties.subnets[0].id 
          : resourceId(vnetResourceGroupName, 'Microsoft.Network/virtualNetworks/subnets', vnetName, subnetName)
      } : null
    }
  }
  dependsOn: [
    vnet
  ]
}

// Model Deployment
resource modelDeployment 'Microsoft.MachineLearningServices/workspaces/models@2023-06-01-preview' = {
  name: '${deploymentName}-model'
  parent: aiFoundryProject
  properties: {
    modelType: 'MLflow'
    modelUri: 'azureml://registries/azureml/models/gpt-4o/versions/1'
    description: 'GPT-4o model deployment'
  }
}

// Deployment Endpoint 
resource endpoint 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints@2023-06-01-preview' = {
  name: '${deploymentName}-endpoint'
  parent: aiFoundryProject
  location: location
  properties: {
    authMode: 'Key'
    traffic: {
      '${deploymentName}': 100
    }
  }
}

// Deployment
resource deployment 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints/deployments@2023-06-01-preview' = {
  name: deploymentName
  parent: endpoint
  location: location
  properties: {
    model: {
      id: modelDeployment.id
    }
    endpointComputeType: 'Managed'
    scaleSettings: {
      scaleType: 'Default'
      instanceCount: 1
      minInstances: 1
      maxInstances: nodeCount
    }
    livenessProbe: {
      failureThreshold: 30
      initialDelaySeconds: 300
      periodSeconds: 10
      successThreshold: 1
      timeoutSeconds: 10
    }
    environmentId: 'azureml://registries/azureml/environments/gpt-4o-runtime/versions/1'
    compute: computeClusterName
    requestSettings: {
      maxQueueWait: 'PT60S'
      maxConcurrentRequestsPerInstance: 5
    }
    environmentVariables: {
      MODEL_MOUNT_PATH: '/mnt/models'
      INFERENCE_SERVER_PORT: '8080'
    }
  }
}

// Outputs
output storageAccountName string = storageAccount.name
output acrName string = acr.name
output aiFoundryProjectName string = aiFoundryProject.name
output endpointUrl string = 'https://${endpoint.name}.inference.ml.azure.com/score'
output computeClusterName string = computeCluster.name
