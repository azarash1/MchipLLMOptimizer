#!/usr/bin/env python3
"""
Utility functions for the LLM Optimizer application.
"""

import os
import sys
import platform
import subprocess
import logging
import psutil
import time
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

def is_apple_silicon() -> bool:
    """Check if the system is running on Apple Silicon"""
    try:
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    except Exception as e:
        logger.error(f'Failed to check Apple Silicon: {str(e)}')
        raise RuntimeError('Failed to check Apple Silicon') from e

def get_system_info() -> Dict[str, Any]:
    """Get information about the system"""
    try:
        memory = psutil.virtual_memory()
        
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "total_memory": memory.total,
            "available_memory": memory.available,
            "memory_percent": memory.percent,
        }
    except Exception as e:
        logger.error(f'Failed to get system info: {str(e)}')
        raise RuntimeError('Failed to get system info') from e

def format_size(size_bytes: int) -> str:
    """Format a size in bytes to a human-readable string"""
    try:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    except Exception as e:
        logger.error(f'Failed to format size: {str(e)}')
        raise RuntimeError('Failed to format size') from e

def measure_memory_usage(func):
    """Decorator to measure memory usage of a function"""
    def wrapper(*args, **kwargs):
        try:
            # Get memory usage before
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss
            
            # Run the function
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Get memory usage after
            mem_after = process.memory_info().rss
            
            # Log the results
            logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: before={format_size(mem_before)}, after={format_size(mem_after)}, diff={format_size(mem_after - mem_before)}")
            
            return result
        except Exception as e:
            logger.error(f'Failed to measure memory usage: {str(e)}')
            raise RuntimeError('Failed to measure memory usage') from e
    return wrapper

def check_gpu_availability() -> bool:
    """Check if GPU is available for MLX"""
    try:
        import mlx.core as mx
        return mx.default_device().platform == "metal"
    except Exception as e:
        logger.error(f'Failed to check GPU availability: {str(e)}')
        raise RuntimeError('Failed to check GPU availability') from e

def estimate_model_size(model_params: int, precision: str = "float32") -> int:
    """
    Estimate the memory size of a model based on its parameters and precision
    
    Args:
        model_params: Number of parameters in the model
        precision: Precision of the parameters (float32, float16, int8, etc.)
        
    Returns:
        Estimated size in bytes
    """
    try:
        bytes_per_param = {
            "float32": 4,
            "float16": 2,
            "int8": 1,
        }
        
        if precision not in bytes_per_param:
            logger.warning(f"Unknown precision {precision}, using float32")
            precision = "float32"
        
        return model_params * bytes_per_param[precision]
    except Exception as e:
        logger.error(f'Failed to estimate model size: {str(e)}')
        raise RuntimeError('Failed to estimate model size') from e

def convert_to_mlx_format(model_dict: Dict[str, Any], dtype: mx.Dtype = mx.float16) -> Dict[str, mx.array]:
    if not model_dict:
        raise ValueError('Input model_dict cannot be empty')
    try:
        # Existing conversion logic
        pass
    except Exception as e:
        logger.error(f'Conversion failed: {str(e)}')
        raise RuntimeError('Model conversion failed') from e
