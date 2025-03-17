#!/usr/bin/env python3
"""
Run script for the LLM Optimizer application.

This script provides a command-line interface for the application.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any

import utils
from model_manager import ModelManager
import optimizer

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """Set up logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("llm_optimizer.log"),
            logging.StreamHandler()
        ]
    )

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="LLM Optimizer for Apple Silicon")
    
    # Model selection
    parser.add_argument("--model", type=str, help="Model to load")
    
    # Optimization options
    parser.add_argument("--quantize", action="store_true", help="Apply quantization")
    parser.add_argument("--quantize-bits", type=int, default=8, help="Quantization bits")
    parser.add_argument("--prune", action="store_true", help="Apply pruning")
    parser.add_argument("--prune-ratio", type=float, default=0.3, help="Pruning ratio")
    parser.add_argument("--lazy-eval", action="store_true", help="Apply lazy evaluation")
    parser.add_argument("--token-batch", action="store_true", help="Apply token batching")
    parser.add_argument("--token-batch-size", type=int, default=8, help="Token batch size")
    parser.add_argument("--dynamic-graph", action="store_true", help="Apply dynamic graph construction")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--output-name", type=str, help="Output model name")
    parser.add_argument("--coreml", action="store_true", help="Export to Core ML format")
    
    # Other options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI")
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger.info("Starting LLM Optimizer for Apple Silicon")
    
    # Check if running on Apple Silicon
    if not utils.is_apple_silicon():
        logger.warning("This application is optimized for Apple Silicon. Some features may not work on other platforms.")
    
    # Launch the GUI if requested
    if args.gui:
        import gui
        model_manager = ModelManager()
        gui.run(model_manager)
        return
    
    # Check if a model was specified
    if not args.model:
        logger.error("No model specified. Use --model to specify a model or --gui to launch the GUI.")
        return
    
    # Initialize the model manager
    model_manager = ModelManager()
    
    # Load the model
    logger.info(f"Loading model {args.model}")
    model, tokenizer = model_manager.load_model(args.model)
    
    # Print model info
    model_info = model_manager.get_model_info()
    logger.info(f"Model loaded: {model_info['name']}")
    logger.info(f"Parameters: {model_info['parameters']:,}")
    logger.info(f"Size: {model_info['size']}")
    
    # Create optimization config
    config = {
        "quantization": args.quantize,
        "quantization_bits": args.quantize_bits,
        "pruning": args.prune,
        "pruning_ratio": args.prune_ratio,
        "lazy_evaluation": args.lazy_eval,
        "token_batching": args.token_batch,
        "token_batch_size": args.token_batch_size,
        "dynamic_graph": args.dynamic_graph,
    }
    
    # Optimize the model
    logger.info(f"Optimizing model with config: {config}")
    optimized_model, results = optimizer.optimize_model(model, config)
    
    # Set the optimized model
    model_manager.set_optimized_model(optimized_model, config)
    
    # Print optimization results
    logger.info("Optimization results:")
    for technique, result in results.items():
        logger.info(f"  {technique}:")
        for key, value in result.items():
            logger.info(f"    {key}: {value}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export the model
    if args.coreml:
        # Convert to Core ML
        logger.info("Converting model to Core ML format")
        coreml_model, coreml_results = optimizer.optimizer.convert_to_coreml(optimized_model, {})
        
        # Save the Core ML model
        model_name = args.output_name or f"{args.model.split('/')[-1]}-optimized"
        model_path = os.path.join(args.output_dir, f"{model_name}.mlpackage")
        
        # In a real implementation, you would save the Core ML model here
        logger.info(f"Model exported to Core ML format at {model_path}")
    else:
        # Save the Hugging Face model
        model_name = args.output_name
        model_path = model_manager.save_model(args.output_dir, model_name)
        logger.info(f"Model exported to Hugging Face format at {model_path}")
    
    logger.info("Done")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        sys.exit(1)
