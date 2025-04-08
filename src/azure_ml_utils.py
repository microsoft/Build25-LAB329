"""
Utility functions for interacting with Azure Machine Learning and Microsoft Azure AI Foundry.
"""

import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, Data
from azure.ai.ml.entities import Environment, BuildContext
from azure.ai.ml.entities import JobInput
from azure.identity import DefaultAzureCredential
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import (
    AZURE_ML_SUBSCRIPTION_ID,
    AZURE_ML_RESOURCE_GROUP,
    AZURE_ML_WORKSPACE_NAME
)

def get_ml_client():
    """
    Get an Azure ML client using default credentials.
    
    Returns:
        MLClient: Azure ML client
    """
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=AZURE_ML_SUBSCRIPTION_ID,
            resource_group_name=AZURE_ML_RESOURCE_GROUP,
            workspace_name=AZURE_ML_WORKSPACE_NAME
        )
        return ml_client
    except Exception as e:
        print(f"Error creating ML client: {e}")
        return None

def register_dataset(ml_client, dataset_path, dataset_name, dataset_description="Dataset for distillation"):
    """
    Register a dataset in Azure ML.
    
    Args:
        ml_client: Azure ML client
        dataset_path: Path to the dataset
        dataset_name: Name for the registered dataset
        dataset_description: Description for the dataset
        
    Returns:
        Registered dataset object
    """
    try:
        dataset = Data(
            path=dataset_path,
            type="uri_folder",
            description=dataset_description,
            name=dataset_name,
            version="1.0.0"
        )
        
        ml_client.data.create_or_update(dataset)
        return dataset
    except Exception as e:
        print(f"Error registering dataset: {e}")
        return None

def create_distillation_job(
    ml_client,
    dataset_name,
    experiment_name,
    source_directory,
    entry_script="distillation_train.py",
    compute_target="cpu-cluster"
):
    """
    Create and submit a distillation job to Azure ML.
    
    Args:
        ml_client: Azure ML client
        dataset_name: Name of the dataset to use
        experiment_name: Name of the experiment
        source_directory: Directory containing training scripts
        entry_script: Main training script
        compute_target: Compute target name
        
    Returns:
        Submitted job object
    """
    try:
        # Define environment with pytorch and transformers
        env = Environment(
            name="distillation-env",
            description="Environment for model distillation",
            build=BuildContext(path="./")
        )
        
        # Define the command
        command = f"python {entry_script} --dataset {dataset_name}"
        
        # Create the job
        from azure.ai.ml import command
        
        job = command(
            code=source_directory,
            command=command,
            environment=env,
            compute=compute_target,
            experiment_name=experiment_name,
            display_name="distillation-training"
        )
        
        # Submit the job
        returned_job = ml_client.jobs.create_or_update(job)
        print(f"Job submitted with ID: {returned_job.id}")
        return returned_job
    
    except Exception as e:
        print(f"Error creating distillation job: {e}")
        return None

def register_distilled_model(ml_client, model_path, model_name, model_version="1.0.0"):
    """
    Register a distilled model in Azure ML.
    
    Args:
        ml_client: Azure ML client
        model_path: Path to the model files
        model_name: Name for the registered model
        model_version: Version of the model
        
    Returns:
        Registered model object
    """
    try:
        model = Model(
            path=model_path,
            name=model_name,
            version=model_version,
            description="Distilled model from GPT-4o"
        )
        
        registered_model = ml_client.models.create_or_update(model)
        print(f"Model registered: {registered_model.name}, version: {registered_model.version}")
        return registered_model
    
    except Exception as e:
        print(f"Error registering model: {e}")
        return None