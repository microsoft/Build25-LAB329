"""
Unit tests for distill_llama_to_phi.py main script that handles distillation from
Llama-4-Scout-17B-16E (teacher model) to Phi-4 (student model).
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.distill_llama_to_phi import parse_args, main

class TestDistillLlamaToPhi(unittest.TestCase):    def test_parse_args_default(self):
        """Test argument parsing with default values."""
        # Test with no arguments
        with patch('sys.argv', ['distill_llama_to_phi.py']):
            args = parse_args()
            self.assertEqual(args.num_examples, 100)
            self.assertEqual(args.student_model, "Phi-4")
            self.assertEqual(args.compute_target, "gpu-cluster")
      def test_parse_args_custom(self):
        """Test argument parsing with custom values."""
        # Test with custom arguments
        with patch('sys.argv', [
            'distill_llama_to_phi.py',
            '--num_examples', '50',
            '--student_model', 'Phi-4-experimental',
            '--compute_target', 'cpu-cluster'
        ]):
            args = parse_args()
            self.assertEqual(args.num_examples, 50)
            self.assertEqual(args.student_model, "Phi-4-experimental")
            self.assertEqual(args.compute_target, "cpu-cluster")
    
    @patch('src.distill_llama_to_phi.initialize_openai_client')
    @patch('src.distill_llama_to_phi.create_sample_prompts')
    @patch('src.distill_llama_to_phi.generate_training_examples')
    @patch('src.distill_llama_to_phi.save_examples_to_jsonl')
    @patch('src.distill_llama_to_phi.get_ml_client')
    @patch('src.distill_llama_to_phi.register_dataset')
    @patch('src.distill_llama_to_phi.create_distillation_job')
    @patch('src.distill_llama_to_phi.load_dotenv')
    def test_main_successful_flow(self, mock_load_dotenv, mock_create_job, 
                                  mock_register_dataset, mock_get_ml_client, 
                                  mock_save_examples, mock_generate_examples,
                                  mock_create_prompts, mock_init_openai):
        """Test the main function with a successful flow."""
        # Setup mocks
        mock_openai_client = MagicMock()
        mock_init_openai.return_value = mock_openai_client
        
        mock_prompts = ["Prompt 1", "Prompt 2"]
        mock_create_prompts.return_value = mock_prompts
        
        mock_examples = [("Prompt 1", "Response 1"), ("Prompt 2", "Response 2")]
        mock_generate_examples.return_value = mock_examples
        
        mock_ml_client = MagicMock()
        mock_get_ml_client.return_value = mock_ml_client
        
        mock_dataset = MagicMock()
        mock_register_dataset.return_value = mock_dataset
        
        mock_job = MagicMock()
        mock_create_job.return_value = mock_job
        
        # Mock command line arguments
        with patch('sys.argv', ['distill_llama_to_phi.py', '--num_examples', '10']):
            # Call the function
            main()
            
            # Verify the workflow
            mock_load_dotenv.assert_called_once()
            mock_init_openai.assert_called_once()
            mock_create_prompts.assert_called_once_with(10)
            mock_generate_examples.assert_called_once_with(mock_openai_client, mock_prompts)
            mock_save_examples.assert_called_once()
            mock_get_ml_client.assert_called_once()
            mock_register_dataset.assert_called_once()
            mock_create_job.assert_called_once()
    
    @patch('src.distill_llama_to_phi.initialize_openai_client')
    def test_main_openai_client_failure(self, mock_init_openai):
        """Test the main function when OpenAI client initialization fails."""
        # Setup mock to return None (failed initialization)
        mock_init_openai.return_value = None
        
        # Mock command line arguments
        with patch('sys.argv', ['distill_llama_to_phi.py']):
            # Call the function
            main()
            
            # Only openai client initialization should be called
            mock_init_openai.assert_called_once()
    
    @patch('src.distill_llama_to_phi.initialize_openai_client')
    @patch('src.distill_llama_to_phi.create_sample_prompts')
    @patch('src.distill_llama_to_phi.generate_training_examples')
    def test_main_no_examples_generated(self, mock_generate_examples, mock_create_prompts, mock_init_openai):
        """Test the main function when no examples are generated."""
        # Setup mocks
        mock_openai_client = MagicMock()
        mock_init_openai.return_value = mock_openai_client
        
        mock_prompts = ["Prompt 1", "Prompt 2"]
        mock_create_prompts.return_value = mock_prompts
        
        # Return empty examples list (failed generation)
        mock_generate_examples.return_value = []
        
        # Mock command line arguments
        with patch('sys.argv', ['distill_llama_to_phi.py']):
            # Call the function
            main()
            
            # Verify client and prompts were created but process stopped after example generation
            mock_init_openai.assert_called_once()
            mock_create_prompts.assert_called_once()
            mock_generate_examples.assert_called_once()

if __name__ == '__main__':
    unittest.main()
