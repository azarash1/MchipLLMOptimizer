# LLM Optimizer for Apple Silicon

This application optimizes Large Language Models (LLMs) for Apple Silicon devices using the MLX framework. It incorporates the latest research in model quantization, pruning, and memory management to enable users to run larger models on lower-spec laptops.

## Features

- **Model Quantization**: Reduce model precision from 32-bit to 16-bit or 8-bit to decrease memory usage
- **Model Pruning**: Remove less important weights to reduce model size
- **Lazy Evaluation**: Defer computations until needed to optimize memory usage
- **Token Batching**: Process tokens in batches for efficient inference
- **Dynamic Graph Construction**: Adapt model shapes without recompilation
- **Unified Memory Utilization**: Leverage Apple Silicon's unified memory architecture

## Requirements

- macOS with Apple Silicon (M1, M2, or M3 chip)
- Python 3.8+
- MLX framework
- Transformers library

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```
python main.py
```

The GUI will guide you through the process of:
1. Selecting a model
2. Choosing optimization techniques
3. Setting optimization parameters
4. Running the optimization
5. Testing the optimized model
6. Saving the optimized model

## Technical Details

This application leverages the latest research in LLM optimization, including techniques from:
- "OpenELM: An Efficient Language Model Family with Open-source Training and Inference Framework" (2024)
- "Beyond Language: Applying MLX Transformers to Engineering Physics" (2024)

It uses Apple's MLX framework, which is specifically designed for efficient machine learning on Apple Silicon devices.

## License

MIT
