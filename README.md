## Build 2025
## Lab329 Fine-Tune End-to-End Distillation Models with Azure OpenAI Service and Azure AI Foundry Project Structure

The solution consists of the following components:

1. Requirements File (requirements.txt): Contains all necessary dependencies for the project.

1. Configuration Module (config/config.py): Stores all configuration parameters including Azure ML workspace details, Azure OpenAI API credentials, and training hyperparameters.

1. Utility Modules:

    1. openai_utils.py: Handles interactions with the Azure OpenAI API to generate responses from the GPT-4o teacher model.
    1. azure_ml_utils.py: Provides functionality to interact with Azure ML and AI Foundry services.
    1. data_utils.py: Contains utilities for data processing, dataset creation, and data loading.
    1. Training Script (src/distillation_train.py): The core training script that runs in Azure ML to perform the actual distillation process.
    1. Main Orchestration Script (src/distill_gpt4o.py): Orchestrates the entire distillation workflow.

### How the Solution Works
The solution implements knowledge distillation, where a smaller, more efficient model (student) learns to mimic the behavior of a larger, more powerful model (GPT-4o as the teacher). Here's the workflow:

1. Generate Training Data: The solution uses GPT-4o to generate responses for a set of prompts, creating teacher-student training pairs.

1. Prepare and Register Dataset: The generated training examples are saved and registered as a dataset in Azure ML.

1. Submit Training Job: A distillation job is created and submitted to Azure AI Foundry, which trains the student model to mimic GPT-4o's responses.

1. Monitor and Register Model: The training job is monitored, and upon completion, the distilled model is registered in Azure ML.

### Using the Solution
To use this solution:

1. First, update the config.py file with your actual Azure credentials and settings.
1. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Run the main distillation script:
```
python src/distill_gpt4o.py --num_examples 200 --student_model distilgpt2 --compute_target your-compute-target
```
4. Monitor the job in the Azure AI Foundry portal.

5. Once completed, you can access and use your distilled model in your applications.

### Customization Options
You can customize this solution by:

- Changing the student model architecture (default is distilgpt2)
- Modifying the sample prompts in data_utils.py to better fit your specific domain
- Adjusting the training hyperparameters in config.py
- Extending the distillation process with advanced techniques like temperature scaling or additional loss functions

This solution provides a complete framework for distilling knowledge from GPT-4o into smaller, more efficient models using Azure AI Foundry's infrastructure.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
