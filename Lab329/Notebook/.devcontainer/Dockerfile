FROM mcr.microsoft.com/vscode/devcontainers/python:3.10

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and TensorFlow with GPU support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    # Core ML frameworks
    tensorflow==2.15.0 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    # Azure ML and related packages
    azureml-core==1.53.0 \
    azureml-dataset-runtime==1.53.0 \
    azureml-defaults==1.53.0 \
    azureml-telemetry==1.53.0 \
    azureml-automl-core==1.53.0 \
    # Microsoft Olive for model optimization
    olive-ai \
    optimum \
    # ONNX Runtime for efficient inference
    onnx \
    onnxruntime \
    onnxruntime-genai \
    # Transformers and related libraries
    transformers==4.36.2 \
    tokenizers==0.15.0 \
    peft \
    bitsandbytes \
    # Utility packages
    numpy==1.23.5 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.2 \
    python-dotenv==1.0.0 \
    # Notebook utilities
    ipykernel==6.25.0 \
    ipywidgets==8.0.7 \
    jupyter==1.0.0

# Create a requirements.txt file for additional packages that might be added later
COPY conda.yml /tmp/
RUN python -c "import yaml; open('/tmp/requirements.txt', 'w').write('\n'.join(yaml.safe_load(open('/tmp/conda.yml'))['dependencies'][1]['pip']))"

# Set up workspace directory
WORKDIR /workspace

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
echo "Phi-4 Model Distillation Environment is ready!"\n\
echo "Python version: $(python --version)"\n\
echo "TensorFlow version: $(python -c "import tensorflow as tf; print(tf.__version__)")"\n\
echo "PyTorch version: $(python -c "import torch; print(torch.__version__)")"\n\
exec "$@"' > /usr/local/bin/entrypoint.sh \
&& chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
