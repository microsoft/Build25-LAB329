"""
Unit tests for data_utils module.
"""
import os
import unittest
import tempfile
import json
from transformers import AutoTokenizer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import (
    DistillationDataset,
    prepare_dataset_from_examples,
    save_examples_to_jsonl,
    load_examples_from_jsonl,
    create_sample_prompts
)

class TestDataUtils(unittest.TestCase):
    def setUp(self):
        self.example_data = [
            ("What is machine learning?", "Machine learning is a branch of artificial intelligence..."),
            ("Explain quantum computing", "Quantum computing uses quantum mechanics to process information...")
        ]
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def test_distillation_dataset(self):
        """Test the DistillationDataset class."""
        dataset = DistillationDataset(self.example_data, self.tokenizer, max_length=128)
        
        # Check dataset length
        self.assertEqual(len(dataset), len(self.example_data))
        
        # Check item structure
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
        
        # Check shapes
        self.assertEqual(item['input_ids'].size(0), 128)
        self.assertEqual(item['attention_mask'].size(0), 128)
        self.assertEqual(item['labels'].size(0), 128)

    def test_prepare_dataset_from_examples(self):
        """Test preparing a dataset from examples."""
        dataloader = prepare_dataset_from_examples(
            self.example_data, self.tokenizer, max_length=128, batch_size=2
        )
        
        # Check that the dataloader has the correct batch size
        self.assertEqual(dataloader.batch_size, 2)
        
        # Check that we can iterate over the dataloader
        batch = next(iter(dataloader))
        self.assertIn('input_ids', batch)
        self.assertIn('attention_mask', batch)
        self.assertIn('labels', batch)

    def test_save_and_load_jsonl(self):
        """Test saving and loading examples to/from JSONL file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "test_data.jsonl")
            
            # Test saving
            save_examples_to_jsonl(self.example_data, temp_file)
            self.assertTrue(os.path.exists(temp_file))
            
            # Check file content
            with open(temp_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), len(self.example_data))
                
                # Check first example
                first_example = json.loads(lines[0])
                self.assertEqual(first_example["prompt"], self.example_data[0][0])
                self.assertEqual(first_example["response"], self.example_data[0][1])
            
            # Test loading
            loaded_examples = load_examples_from_jsonl(temp_file)
            self.assertEqual(len(loaded_examples), len(self.example_data))
            self.assertEqual(loaded_examples[0][0], self.example_data[0][0])
            self.assertEqual(loaded_examples[0][1], self.example_data[0][1])

    def test_create_sample_prompts(self):
        """Test creating sample prompts."""
        # Test with default number
        prompts = create_sample_prompts()
        self.assertGreaterEqual(len(prompts), 10)  # Function has at least 10 default prompts
        
        # Test with custom number
        custom_num = 15
        prompts = create_sample_prompts(custom_num)
        self.assertEqual(len(prompts), custom_num)

if __name__ == '__main__':
    unittest.main()