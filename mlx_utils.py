#!/usr/bin/env python3
"""
MLX utilities for the LLM Optimizer application.

This module provides utilities for working with the MLX framework.
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import time

logger = logging.getLogger(__name__)

def convert_to_mlx_format(model_dict: Dict[str, Any], dtype: mx.Dtype = mx.float16) -> Dict[str, mx.array]:
    """
    Convert a PyTorch model dictionary to MLX format with optional dtype conversion
    
    Args:
        model_dict: PyTorch model state dictionary
        dtype: Target data type (default: float16 for mixed precision)
        
    Returns:
        MLX model state dictionary
    """
    mlx_dict = {}
    
    for key, value in model_dict.items():
        # Convert PyTorch tensor to NumPy array
        numpy_array = value.detach().cpu().numpy()
        
        # Convert NumPy array to MLX array with specified dtype
        mlx_array = mx.array(numpy_array, dtype=dtype)
        
        # Store in the MLX dictionary
        mlx_dict[key] = mlx_array
    
    return mlx_dict

def quantize_weights_block_wise(weights: mx.array, block_size: int = 32) -> Tuple[mx.array, mx.array]:
    shape = weights.shape
    blocks = weights.reshape(-1, block_size)
    
    # Calculate scale factors per block
    scales = mx.max(mx.abs(blocks), axis=1) / 7.5
    scales = mx.expand_dims(scales, axis=1)
    
    # Quantize with 4-bit precision
    quantized = mx.clip(mx.round(blocks / scales), -7, 7)
    return quantized.reshape(shape), scales

def dequantize_weights(quantized: mx.array, scales: mx.array, original_shape: Tuple[int]) -> mx.array:
    return (quantized * scales).reshape(original_shape)

def optimize_attention(model: nn.Module, use_fused_sdpa: bool = True) -> nn.Module:
    """
    Optimize attention computation using fused SDPA (Scaled Dot Product Attention)
    Based on Apple's MLX optimization research
    
    Args:
        model: MLX model
        use_fused_sdpa: Whether to use fused SDPA optimization
        
    Returns:
        Model with optimized attention
    """
    if not use_fused_sdpa:
        return model
    
    # Enable fused attention computation
    model.config.use_fused_attention = True
    
    # Set optimal attention algorithm
    model.config.attention_algorithm = "flash"
    
    return model

def setup_kv_cache(model: nn.Module, method: str = "flexible", window_size: int = 2048) -> nn.Module:
    """
    Set up KV cache with flexible shaped inputs
    Based on Apple's research for optimal memory usage
    
    Args:
        model: MLX model
        method: Cache method ('flexible' or 'fixed')
        window_size: Size of the cache window
        
    Returns:
        Model with KV cache optimization
    """
    # Enable KV cache
    model.config.use_cache = True
    
    if method == "flexible":
        # Use flexible shaped inputs for better memory efficiency
        model.config.use_flexible_shapes = True
        model.config.max_position_embeddings = window_size
    
    return model

def warm_up_model(model: nn.Module, tokenizer: Any, num_tokens: int = 1024) -> None:
    """
    Warm up a model with a representative workload
    Based on Apple's MLX optimization research
    
    Args:
        model: MLX model
        tokenizer: Tokenizer for the model
        num_tokens: Number of tokens to use for warm-up
    """
    # Create a more representative input
    sample_text = "This is a sample text for warming up the model. " * (num_tokens // 32)
    inputs = tokenizer(sample_text, return_tensors="pt", max_length=num_tokens, truncation=True)
    
    # Convert to MLX array
    inputs = {k: mx.array(v.numpy()) for k, v in inputs.items()}
    
    # Run multiple forward passes with different lengths
    lengths = [64, 128, 256, 512, 1024]
    for length in lengths:
        if length > num_tokens:
            continue
        truncated_inputs = {k: v[:, :length] for k, v in inputs.items()}
        _ = model(truncated_inputs)
    
    logger.info(f"Model warmed up with {num_tokens} tokens at various sequence lengths")

def benchmark_model(model: nn.Module, tokenizer: Any, prompt: str, num_tokens: int = 100) -> Dict[str, Any]:
    """
    Comprehensive model benchmarking
    Based on Apple's MLX performance measurement methodology
    
    Args:
        model: MLX model
        tokenizer: Tokenizer for the model
        prompt: Prompt to use for benchmarking
        num_tokens: Number of tokens to generate
        
    Returns:
        Dictionary of benchmark results
    """
    if not prompt or len(prompt) < 10:
        raise ValueError('Prompt must be at least 10 characters')
    if num_tokens < 1 or num_tokens > 1000:
        raise ValueError('num_tokens must be between 1-1000')
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = len(inputs["input_ids"][0])
    
    # Convert to MLX array
    inputs = {k: mx.array(v.numpy()) for k, v in inputs.items()}
    
    results = {}
    
    # 1. Measure TTFT (Time to First Token)
    start_time = time.time()
    outputs = model(inputs)
    first_token = model.generate(inputs, max_length=input_len + 1)
    ttft = time.time() - start_time
    results["ttft_ms"] = ttft * 1000
    
    # 2. Measure extend throughput
    start_time = time.time()
    generated = model.generate(
        inputs,
        max_length=input_len + num_tokens,
        do_sample=True,
        temperature=0.7
    )
    total_time = time.time() - start_time
    
    # Calculate throughput
    tokens_generated = len(generated[0]) - input_len
    throughput = tokens_generated / total_time
    results["extend_throughput"] = {
        "tokens_per_second": throughput,
        "total_tokens": tokens_generated,
        "total_time_seconds": total_time
    }
    
    # 3. Memory usage
    results["memory_usage"] = {
        "peak_memory_mb": mx.memory.peak_memory() / (1024 * 1024),
        "current_memory_mb": mx.memory.current_memory() / (1024 * 1024)
    }
    
    # Add latency measurement
    latencies = []
    for _ in range(10):
        start = time.time()
        _ = model.generate(inputs, max_length=1)
        latencies.append(time.time() - start)
    
    # Calculate percentiles
    results["latency_p90"] = np.percentile(latencies, 90) * 1000
    
    return results

def quantize_model(model: nn.Module) -> nn.Module:
    for name, weights in model.named_parameters():
        quantized, scales = quantize_weights_block_wise(weights)
        model.register_parameter(f'{name}_quantized', quantized)
        model.register_buffer(f'{name}_scales', scales)
    return model
