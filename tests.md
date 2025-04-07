Unit tests for your DistillationGPT4o project.

## Test Files:

- test_data_utils.py: Tests for dataset handling, JSONL file operations, and data preparation
- test_openai_utils.py: Tests for OpenAI client initialization and response generation
- test_azure_ml_utils.py: Tests for Azure ML operations like client initialization, dataset and model registration
- test_distill_gpt4o.py: Tests for main script functionality and command-line argument parsing
- test_distillation_train.py: Tests for the training process and its components
- test_local_gpu_train.py:  Test for the local GPU implementation

### To Run Tests
- Run all tests: python run_tests.py
- Run specific tests: python [run_tests.py](http://_vscodecontentref_/11) data (runs tests with "data" in the name)
- Run tests with verbose output: python [run_tests.py](http://_vscodecontentref_/12) -v
- All tests are designed with proper mocking to avoid actual API calls or real model training during testing. The tests cover positive cases (successful execution) as well as error handling and edge cases.

To run the tests, you can use the provided run_tests.py script:
```
python run_tests.py
```
