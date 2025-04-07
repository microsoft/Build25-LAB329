"""
Unit tests for openai_utils module.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.openai_utils import (
    initialize_openai_client,
    generate_response,
    generate_training_examples
)

class TestOpenAIUtils(unittest.TestCase):
    @patch('src.openai_utils.openai')
    def test_initialize_openai_client(self, mock_openai):
        """Test initializing the Azure OpenAI client."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.AzureOpenAI.return_value = mock_client

        # Call the function
        client = initialize_openai_client()
        
        # Assertions
        self.assertIsNotNone(client)
        mock_openai.AzureOpenAI.assert_called_once()
        self.assertEqual(mock_openai.api_type, "azure")
        self.assertIsNotNone(mock_openai.api_key)
        self.assertIsNotNone(mock_openai.api_base)
        self.assertIsNotNone(mock_openai.api_version)

    @patch('src.openai_utils.generate_response')
    def test_generate_training_examples(self, mock_generate_response):
        """Test generating training examples."""
        # Setup mock
        mock_client = MagicMock()
        mock_generate_response.side_effect = [
            "Response 1", 
            "Response 2", 
            None  # Simulate a failed response
        ]
        
        # Test data
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        
        # Call the function
        examples = generate_training_examples(mock_client, prompts)
        
        # Assertions
        self.assertEqual(len(examples), 2)  # Only 2 successful responses
        self.assertEqual(examples[0][0], "Prompt 1")
        self.assertEqual(examples[0][1], "Response 1")
        self.assertEqual(examples[1][0], "Prompt 2")
        self.assertEqual(examples[1][1], "Response 2")
        
        # Check generate_response was called correctly
        self.assertEqual(mock_generate_response.call_count, 3)

    @patch('src.openai_utils.openai.AzureOpenAI')
    def test_generate_response(self, mock_azure_openai):
        """Test generating a response from the model."""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content.strip.return_value = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        # Call the function
        response = generate_response(mock_client, "Test prompt", max_tokens=50)
        
        # Assertions
        self.assertEqual(response, "Test response")
        mock_client.chat.completions.create.assert_called_once()
        
    @patch('src.openai_utils.openai.AzureOpenAI')
    def test_generate_response_error(self, mock_azure_openai):
        """Test error handling in generate_response."""
        # Setup mock to raise an exception
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Call the function
        response = generate_response(mock_client, "Test prompt")
        
        # Assertions
        self.assertIsNone(response)
        mock_client.chat.completions.create.assert_called_once()

if __name__ == '__main__':
    unittest.main()