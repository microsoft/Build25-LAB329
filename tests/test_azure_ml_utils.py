"""
Unit tests for azure_ml_utils module.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.azure_ml_utils import (
    get_ml_client,
    register_dataset,
    create_distillation_job,
    register_distilled_model
)

class TestAzureMLUtils(unittest.TestCase):
    @patch('src.azure_ml_utils.DefaultAzureCredential')
    @patch('src.azure_ml_utils.MLClient')
    def test_get_ml_client_success(self, mock_ml_client, mock_credential):
        """Test successfully getting an Azure ML client."""
        # Setup mock
        mock_credential_instance = MagicMock()
        mock_credential.return_value = mock_credential_instance
        
        mock_client_instance = MagicMock()
        mock_ml_client.return_value = mock_client_instance
        
        # Call the function
        client = get_ml_client()
        
        # Assertions
        self.assertIsNotNone(client)
        mock_credential.assert_called_once()
        mock_ml_client.assert_called_once_with(
            credential=mock_credential_instance,
            subscription_id=unittest.mock.ANY,
            resource_group_name=unittest.mock.ANY,
            workspace_name=unittest.mock.ANY
        )
    
    @patch('src.azure_ml_utils.DefaultAzureCredential')
    @patch('src.azure_ml_utils.MLClient')
    def test_get_ml_client_error(self, mock_ml_client, mock_credential):
        """Test error handling when getting an Azure ML client."""
        # Setup mock to raise an exception
        mock_ml_client.side_effect = Exception("Connection Error")
        
        # Call the function
        client = get_ml_client()
        
        # Assertions
        self.assertIsNone(client)
    
    @patch('src.azure_ml_utils.Data')
    def test_register_dataset_success(self, mock_data):
        """Test successfully registering a dataset."""
        # Setup mocks
        mock_ml_client = MagicMock()
        mock_data_instance = MagicMock()
        mock_data.return_value = mock_data_instance
        
        # Call the function
        dataset = register_dataset(
            mock_ml_client, 
            "test/path", 
            "test-dataset", 
            "Test description"
        )
        
        # Assertions
        self.assertIsNotNone(dataset)
        mock_data.assert_called_once_with(
            path="test/path",
            type="uri_folder",
            description="Test description",
            name="test-dataset",
            version="1.0.0"
        )
        mock_ml_client.data.create_or_update.assert_called_once_with(mock_data_instance)
    
    @patch('src.azure_ml_utils.Data')
    def test_register_dataset_error(self, mock_data):
        """Test error handling when registering a dataset."""
        # Setup mocks
        mock_ml_client = MagicMock()
        mock_ml_client.data.create_or_update.side_effect = Exception("Registration Error")
        mock_data_instance = MagicMock()
        mock_data.return_value = mock_data_instance
        
        # Call the function
        dataset = register_dataset(
            mock_ml_client, 
            "test/path", 
            "test-dataset"
        )
        
        # Assertions
        self.assertIsNone(dataset)
    
    @patch('src.azure_ml_utils.Environment')
    @patch('src.azure_ml_utils.BuildContext')
    @patch('src.azure_ml_utils.command')
    def test_create_distillation_job_success(self, mock_command, mock_build_context, mock_env):
        """Test successfully creating a distillation job."""
        # Setup mocks
        mock_ml_client = MagicMock()
        mock_env_instance = MagicMock()
        mock_env.return_value = mock_env_instance
        
        mock_build_context_instance = MagicMock()
        mock_build_context.return_value = mock_build_context_instance
        
        mock_job = MagicMock()
        mock_command.return_value = mock_job
        
        mock_returned_job = MagicMock()
        mock_ml_client.jobs.create_or_update.return_value = mock_returned_job
        
        # Call the function
        job = create_distillation_job(
            mock_ml_client,
            "test-dataset",
            "test-experiment",
            "./src",
            "test_script.py",
            "test-cluster"
        )
        
        # Assertions
        self.assertIsNotNone(job)
        mock_env.assert_called_once()
        mock_command.assert_called_once()
        mock_ml_client.jobs.create_or_update.assert_called_once_with(mock_job)
    
    @patch('src.azure_ml_utils.Environment')
    @patch('src.azure_ml_utils.command')
    def test_create_distillation_job_error(self, mock_command, mock_env):
        """Test error handling when creating a distillation job."""
        # Setup mocks
        mock_ml_client = MagicMock()
        mock_ml_client.jobs.create_or_update.side_effect = Exception("Job Creation Error")
        
        # Call the function
        job = create_distillation_job(
            mock_ml_client,
            "test-dataset",
            "test-experiment",
            "./src"
        )
        
        # Assertions
        self.assertIsNone(job)
    
    @patch('src.azure_ml_utils.Model')
    def test_register_distilled_model_success(self, mock_model):
        """Test successfully registering a distilled model."""
        # Setup mocks
        mock_ml_client = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_registered_model = MagicMock()
        mock_ml_client.models.create_or_update.return_value = mock_registered_model
        
        # Call the function
        model = register_distilled_model(
            mock_ml_client,
            "test/model/path",
            "test-model",
            "1.0.0"
        )
        
        # Assertions
        self.assertIsNotNone(model)
        mock_model.assert_called_once_with(
            path="test/model/path",
            name="test-model",
            version="1.0.0",
            description="Distilled model from GPT-4o"
        )
        mock_ml_client.models.create_or_update.assert_called_once_with(mock_model_instance)
    
    @patch('src.azure_ml_utils.Model')
    def test_register_distilled_model_error(self, mock_model):
        """Test error handling when registering a distilled model."""
        # Setup mocks
        mock_ml_client = MagicMock()
        mock_ml_client.models.create_or_update.side_effect = Exception("Model Registration Error")
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Call the function
        model = register_distilled_model(
            mock_ml_client,
            "test/model/path",
            "test-model"
        )
        
        # Assertions
        self.assertIsNone(model)

if __name__ == '__main__':
    unittest.main()