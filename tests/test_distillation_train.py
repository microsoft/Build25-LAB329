"""
Unit tests for distillation_train.py script.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.distillation_train import parse_args, train

class TestDistillationTrain(unittest.TestCase):
    def test_parse_args_default(self):
        """Test argument parsing with default values."""
        # Test with only required argument
        with patch('sys.argv', ['distillation_train.py', '--dataset', 'test-dataset']):
            args = parse_args()
            self.assertEqual(args.dataset, 'test-dataset')
            self.assertEqual(args.student_model, "distilgpt2")
            self.assertEqual(args.output_dir, "./outputs")
    
    def test_parse_args_custom(self):
        """Test argument parsing with custom values."""
        # Test with custom arguments
        with patch('sys.argv', [
            'distillation_train.py',
            '--dataset', 'test-dataset',
            '--student_model', 'gpt2',
            '--output_dir', './custom-outputs'
        ]):
            args = parse_args()
            self.assertEqual(args.dataset, 'test-dataset')
            self.assertEqual(args.student_model, "gpt2")
            self.assertEqual(args.output_dir, "./custom-outputs")
    
    @patch('src.distillation_train.Run')
    @patch('src.distillation_train.AutoTokenizer')
    @patch('src.distillation_train.AutoModelForCausalLM')
    @patch('src.distillation_train.load_examples_from_jsonl')
    @patch('src.distillation_train.prepare_dataset_from_examples')
    @patch('src.distillation_train.torch.optim.AdamW')
    @patch('src.distillation_train.get_linear_schedule_with_warmup')
    @patch('builtins.open')
    @patch('os.makedirs')
    def test_training_process(self, mock_makedirs, mock_open, mock_scheduler, 
                              mock_optimizer, mock_prepare_dataset, mock_load_examples,
                              mock_model_class, mock_tokenizer_class, mock_run):
        """Test the training process with mocked components."""
        # Setup mocks
        mock_run_instance = MagicMock()
        mock_run.get_context.return_value = mock_run_instance
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock examples and dataloader
        mock_examples = [("Prompt 1", "Response 1"), ("Prompt 2", "Response 2")]
        mock_load_examples.return_value = mock_examples
        
        mock_batch = {
            "input_ids": torch.ones((2, 10), dtype=torch.long),
            "attention_mask": torch.ones((2, 10), dtype=torch.long),
            "labels": torch.ones((2, 10), dtype=torch.long)
        }
        mock_dataloader = MagicMock()
        mock_dataloader.__len__.return_value = 2
        mock_dataloader.__iter__.return_value = [mock_batch]
        mock_prepare_dataset.return_value = mock_dataloader
        
        # Mock optimizer and scheduler
        mock_optimizer_instance = MagicMock()
        mock_optimizer.return_value = mock_optimizer_instance
        
        mock_scheduler_instance = MagicMock()
        mock_scheduler.return_value = mock_scheduler_instance
        
        # Mock model outputs
        mock_outputs = MagicMock()
        mock_outputs.loss = torch.tensor(0.5)
        mock_model.return_value = mock_outputs
        
        # Mock device (force CPU for testing)
        with patch('torch.cuda.is_available', return_value=False):
            # Mock command line arguments
            with patch('sys.argv', ['distillation_train.py', '--dataset', 'test-dataset']):
                # Call the function
                train()
                
                # Verify the training flow
                mock_run.get_context.assert_called_once()
                mock_tokenizer_class.from_pretrained.assert_called_once_with("distilgpt2")
                mock_model_class.from_pretrained.assert_called_once_with("distilgpt2")
                
                # Verify pad token assignment
                self.assertEqual(mock_tokenizer.pad_token, mock_tokenizer.eos_token)
                
                # Verify dataset loading
                mock_load_examples.assert_called_once()
                self.assertIn('/tmp/datasets/test-dataset/data.jsonl', mock_load_examples.call_args[0][0])
                
                # Verify dataloader preparation
                mock_prepare_dataset.assert_called_once_with(
                    mock_examples, mock_tokenizer, unittest.mock.ANY, unittest.mock.ANY
                )
                
                # Verify optimizer and scheduler creation
                mock_optimizer.assert_called_once()
                mock_scheduler.assert_called_once()
                
                # Verify model saving
                self.assertEqual(mock_model.save_pretrained.call_count, 2)  # Once for checkpoint, once for final model
                
                # Verify model registration
                mock_run_instance.upload_folder.assert_called_once()
                mock_run_instance.register_model.assert_called_once()

if __name__ == '__main__':
    unittest.main()