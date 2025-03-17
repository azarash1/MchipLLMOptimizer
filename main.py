#!/usr/bin/env python3
"""
LLM Optimizer for Apple Silicon - GUI Application

This is the main entry point for the LLM Optimizer application.
It initializes the GUI and handles the main application flow.
"""

import os
import sys
import logging
from datetime import datetime
import tkinter as tk

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_optimizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try importing required libraries, with helpful error messages if they're missing
try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    logger.error("MLX framework not found. Please install it with: pip install mlx")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    logger.error("Transformers library not found. Please install it with: pip install transformers")
    sys.exit(1)

# Import local modules
from optimizer import ModelOptimizer, OptimizationConfig
import gui
import utils
from model_manager import ModelManager

def main():
    """Main application entry point"""
    logger.info("Starting LLM Optimizer for Apple Silicon")
    
    # Check if running on Apple Silicon
    if not utils.is_apple_silicon():
        logger.warning("This application is optimized for Apple Silicon. Some features may not work on other platforms.")
    
    # Initialize the model manager
    model_manager = ModelManager()
    
    # Create the main window
    root = tk.Tk()
    root.title("LLM Optimizer for Apple Silicon")
    
    # Start the GUI
    gui.run(model_manager)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        sys.exit(1)
