targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the environment that can be used as part of naming resource convention')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
param location string
@metadata({azd: {
  type: 'location'
  usageName: 'AIServices.GlobalStandard.MaaS,1'
  }
})
param deepSeekV31Location string


@description('Id of the user or app to assign application roles')
param principalId string

// Tags that should be applied to all resources.
// 
// Note that 'azd-service-name' tags should be applied separately to service host resources.
// Example usage:
//   tags: union(tags, { 'azd-service-name': <service name in azure.yaml> })
var tags = {
  'azd-env-name': environmentName
}

// Organize resources in a resource group
resource rg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: 'rg-${environmentName}'
  location: location
  tags: tags
}

module resources 'resources.bicep' = {
  scope: rg
  name: 'resources'
  params: {
    location: location
    tags: tags
    principalId: principalId
    aiFoundryProjectConnectionString: aiModelsDeploy.outputs.aiFoundryProjectConnectionString
  }
}

module aiModelsDeploy 'ai-project.bicep' = {
  scope: rg
  name: 'ai-project'
  params: {
    deepSeekV31Location:  deepSeekV31Location    
    tags: tags
    location: location
    envName: environmentName
  }
}
output AZURE_AI_PROJECT_CONNECTION_STRING string = aiModelsDeploy.outputs.aiFoundryProjectConnectionString
output AZURE_RESOURCE_AI_PROJECT_ID string = aiModelsDeploy.outputs.projectId
