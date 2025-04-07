"""
Unit tests for local_gpu_train.py script.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.local_gpu_train import parse_args, train, main

class TestLocalGPUTrain(unittest.TestCase):
    def test_parse_args_default(self):
        """Test argument parsing with default values."""
        # Test with only required argument
        with patch('sys.argv', ['local_gpu_train.py', '--dataset_path', 'data/test-dataset.jsonl']):
            args = parse_args()
            self.assertEqual(args.dataset_path, 'data/test-dataset.jsonl')
            self.assertEqual(args.student_model, "distilgpt2")
            self.assertEqual(args.output_dir, "./outputs")
            self.assertEqual(args.batch_size, 8)  # Default from config
            self.assertEqual(args.fp16, False)
    
    def test_parse_args_custom(self):
        """Test argument parsing with custom values."""
        # Test with custom arguments
        with patch('sys.argv', [
            'local_gpu_train.py',
            '--dataset_path', 'data/test-dataset.jsonl',
            '--student_model', 'gpt2',
            '--output_dir', './custom-outputs',
            '--batch_size', '16',
            '--fp16'
        ]):
            args = parse_args()
            self.assertEqual(args.dataset_path, 'data/test-dataset.jsonl')
            self.assertEqual(args.student_model, "gpt2")
            self.assertEqual(args.output_dir, "./custom-outputs")
            self.assertEqual(args.batch_size, 16)
            self.assertEqual(args.fp16, True)
    
    @patch('src.local_gpu_train.torch.cuda.is_available')
    @patch('src.local_gpu_train.AutoTokenizer')
    @patch('src.local_gpu_train.AutoModelForCausalLM')
    @patch('src.local_gpu_train.load_examples_from_jsonl')
    @patch('src.local_gpu_train.torch.optim.AdamW')
    @patch('src.local_gpu_train.get_linear_schedule_with_warmup')
    @patch('os.makedirs')
    def test_training_process_single_gpu(self, mock_makedirs, mock_scheduler, 
                                        mock_optimizer, mock_load_examples,
                                        mock_model_class, mock_tokenizer_class,
                                        mock_cuda_available):
        """Test the training process with a single GPU."""
        # Setup mocks
        mock_cuda_available.return_value = True
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock examples and dataloader
        mock_examples = [("Prompt 1", "Response 1"), ("Prompt 2", "Response 2")]
        mock_load_examples.return_value = mock_examples
        
        # Mock batch and dataset
        mock_batch = {
            "input_ids": torch.ones((2, 10), dtype=torch.long),
            "attention_mask": torch.ones((2, 10), dtype=torch.long),
            "labels": torch.ones((2, 10), dtype=torch.long)
        }
        
        # Mock DataLoader through DatasetSampler
        with patch('src.local_gpu_train.DistillationDataset') as mock_dataset_class:
            mock_dataset = MagicMock()
            mock_dataset_class.return_value = mock_dataset
            
            with patch('src.local_gpu_train.DataLoader') as mock_dataloader_class:
                mock_dataloader = MagicMock()
                mock_dataloader.__iter__.return_value = [mock_batch]
                mock_dataloader.__len__.return_value = 2
                mock_dataloader_class.return_value = mock_dataloader
                
                # Mock optimizer and scheduler
                mock_optimizer_instance = MagicMock()
                mock_optimizer.return_value = mock_optimizer_instance
                
                mock_scheduler_instance = MagicMock()
                mock_scheduler.return_value = mock_scheduler_instance
                
                # Mock model outputs
                mock_outputs = MagicMock()
                mock_outputs.loss = torch.tensor(0.5)
                mock_model.return_value = mock_outputs
                
                # Create mock args
                args = MagicMock()
                args.dataset_path = "test/dataset.jsonl"
                args.student_model = "distilgpt2"
                args.output_dir = "test/output"
                args.batch_size = 2
                args.learning_rate = 5e-5
                args.num_epochs = 1
                args.max_seq_length = 128
                args.gradient_accumulation_steps = 1
                args.fp16 = False
                args.seed = 42
                
                # Call the function
                train(0, 1, args)
                
                # Verify the training flow
                mock_tokenizer_class.from_pretrained.assert_called_once_with("distilgpt2")
                mock_model_class.from_pretrained.assert_called_once_with("distilgpt2")
                
                # Verify pad token assignment
                self.assertEqual(mock_tokenizer.pad_token, mock_tokenizer.eos_token)
                
                # Verify dataset loading and preprocessing
                mock_load_examples.assert_called_once_with("test/dataset.jsonl")
                mock_dataset_class.assert_called_once_with(mock_examples, mock_tokenizer, max_length=128)
                mock_dataloader_class.assert_called_once()
                
                # Verify optimizer and scheduler creation
                mock_optimizer.assert_called_once()
                mock_scheduler.assert_called_once()
                
                # Verify model saving
                self.assertEqual(mock_model.save_pretrained.call_count, 2)  # Once for checkpoint, once for final model

    @patch('src.local_gpu_train.torch.cuda.device_count')
    @patch('src.local_gpu_train.train')
    @patch('src.local_gpu_train.mp.spawn')
    def test_main_single_gpu(self, mock_spawn, mock_train, mock_device_count):
        """Test the main function with a single GPU."""
        # Setup mocks
        mock_device_count.return_value = 1
        
        # Mock command line arguments
        with patch('sys.argv', ['local_gpu_train.py', '--dataset_path', 'test/dataset.jsonl']):
            # Call the function
            main()
            
            # Verify single GPU path was taken (not using spawn)
            mock_train.assert_called_once()
            mock_spawn.assert_not_called()
    
    @patch('src.local_gpu_train.torch.cuda.device_count')
    @patch('src.local_gpu_train.train')
    @patch('src.local_gpu_train.mp.spawn')
    def test_main_multi_gpu(self, mock_spawn, mock_train, mock_device_count):
        """Test the main function with multiple GPUs."""
        # Setup mocks to simulate 4 GPUs
        mock_device_count.return_value = 4
        
        # Mock command line arguments
        with patch('sys.argv', ['local_gpu_train.py', '--dataset_path', 'test/dataset.jsonl']):
            # Call the function
            main()
            
            # Verify multi-GPU path was taken (using spawn)
            mock_spawn.assert_called_once()
            mock_train.assert_not_called()  # train is called via spawn

if __name__ == '__main__':
    unittest.main()