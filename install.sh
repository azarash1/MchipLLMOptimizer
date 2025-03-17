#!/bin/bash

# Install script for the LLM Optimizer application

echo "Installing LLM Optimizer for Apple Silicon..."

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This application is designed for macOS with Apple Silicon."
    exit 1
fi

# Validate Apple Silicon
if [[ $(uname -m) != 'arm64' ]]; then
    echo 'Error: This script requires Apple Silicon (M1/M2/M3)'
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Please install Python 3 from https://www.python.org/downloads/"
    exit 1
fi

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --prefer-binary

# Make the run script executable
echo "Making run script executable..."
chmod +x run.py

echo "Installation complete!"
echo "To run the application, use:"
echo "  source venv/bin/activate"
echo "  ./run.py --gui"
