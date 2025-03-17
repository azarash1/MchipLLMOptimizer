#!/usr/bin/env python3
"""
Model Manager for the LLM Optimizer application.

This module handles loading, saving, and managing models.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import time

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

import utils

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages models for the LLM Optimizer application"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.model_config = None
        self.optimized_model = None
        self.optimization_config = None
        self.models_cache_dir = os.path.expanduser("~/.cache/llm_optimizer/models")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.models_cache_dir, exist_ok=True)
        
        # Popular models that work well with MLX
        self.recommended_models = [
            "facebook/opt-125m",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "mistralai/Mistral-7B-v0.1",
            "microsoft/phi-1_5",
            "microsoft/phi-2",
            "meta-llama/Llama-2-7b-hf",
        ]
    
    @utils.measure_memory_usage
    def load_model(self, model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a model and tokenizer from Hugging Face
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.models_cache_dir)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=self.models_cache_dir,
            device_map="auto",  # Use the most efficient device mapping
            torch_dtype="auto",  # Use the most efficient dtype
        )
        
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.model_config = model.config.to_dict()
        
        logger.info(f"Model {model_name} loaded successfully")
        return model, tokenizer
    
    def save_model(self, output_dir: str, model_name: str = None) -> str:
        """
        Save the optimized model to disk
        
        Args:
            output_dir: Directory to save the model to
            model_name: Name to save the model as
            
        Returns:
            Path to the saved model
        """
        if self.optimized_model is None:
            raise ValueError("No optimized model to save")
        
        if model_name is None:
            model_name = f"{self.model_name.split('/')[-1]}-optimized"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model path
        model_path = os.path.join(output_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        self.optimized_model.save_pretrained(model_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(model_path)
        
        # Save optimization config
        with open(os.path.join(model_path, "optimization_config.json"), "w") as f:
            json.dump(self.optimization_config, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def convert_to_mlx(self, model: PreTrainedModel) -> nn.Module:
        """
        Convert a Hugging Face model to MLX format
        
        Args:
            model: Hugging Face model to convert
            
        Returns:
            MLX model
        """
        logger.info("Converting model to MLX format")
        
        # This is a placeholder for the actual conversion logic
        # In a real implementation, you would convert the model to MLX format here
        
        # For now, we'll just return the original model
        return model
    
    @utils.measure_memory_usage
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """
        Generate text using the model
        
        Args:
            prompt: Prompt to generate text from
            max_length: Maximum length of the generated text
            temperature: Temperature for sampling
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model loaded")
        
        logger.info(f"Generating text with prompt: {prompt}")
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate text
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
        )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Generated text: {generated_text}")
        return generated_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        # Get model parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        
        # Estimate model size
        model_size_bytes = utils.estimate_model_size(num_params)
        
        return {
            "name": self.model_name,
            "parameters": num_params,
            "size": utils.format_size(model_size_bytes),
            "config": self.model_config,
        }
    
    def set_optimized_model(self, model: PreTrainedModel, config: Dict[str, Any]) -> None:
        """
        Set the optimized model and its configuration
        
        Args:
            model: Optimized model
            config: Optimization configuration
        """
        self.optimized_model = model
        self.optimization_config = config
