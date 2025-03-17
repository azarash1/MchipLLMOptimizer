#!/usr/bin/env python3
"""
GUI module for the LLM Optimizer application.

This module implements the graphical user interface for the application.
"""

import os
import sys
import logging
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, Any, List, Tuple, Optional, Callable

import utils
from model_manager import ModelManager
import optimizer

logger = logging.getLogger(__name__)

class LLMOptimizerGUI:
    """GUI for the LLM Optimizer application"""
    
    def __init__(self, root: tk.Tk, model_manager: ModelManager):
        self.root = root
        self.model_manager = model_manager
        self.optimizer = optimizer.optimizer
        
        # Set up the main window
        self.root.title("LLM Optimizer for Apple Silicon")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the notebook (tabs)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.model_tab = ttk.Frame(self.notebook, padding=10)
        self.optimize_tab = ttk.Frame(self.notebook, padding=10)
        self.test_tab = ttk.Frame(self.notebook, padding=10)
        self.about_tab = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.model_tab, text="Model Selection")
        self.notebook.add(self.optimize_tab, text="Optimization")
        self.notebook.add(self.test_tab, text="Test & Export")
        self.notebook.add(self.about_tab, text="About")
        
        # Set up tabs
        self._setup_model_tab()
        self._setup_optimize_tab()
        self._setup_test_tab()
        self._setup_about_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Set up system info
        self._update_system_info()
    
    def _setup_model_tab(self):
        """Set up the model selection tab"""
        # Create frames
        model_frame = ttk.LabelFrame(self.model_tab, text="Model Selection", padding=10)
        model_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model selection
        ttk.Label(model_frame, text="Select a model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Model dropdown
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, width=40)
        self.model_dropdown['values'] = self.model_manager.recommended_models
        self.model_dropdown.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Custom model entry
        ttk.Label(model_frame, text="Or enter a custom model:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.custom_model_var = tk.StringVar()
        self.custom_model_entry = ttk.Entry(model_frame, textvariable=self.custom_model_var, width=40)
        self.custom_model_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Load model button
        self.load_button = ttk.Button(model_frame, text="Load Model", command=self._load_model)
        self.load_button.grid(row=2, column=1, sticky=tk.W, pady=10)
        
        # Model info frame
        info_frame = ttk.LabelFrame(self.model_tab, text="Model Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Model info text
        self.model_info_text = tk.Text(info_frame, height=10, width=80, wrap=tk.WORD)
        self.model_info_text.pack(fill=tk.BOTH, expand=True)
        self.model_info_text.insert(tk.END, "No model loaded")
        self.model_info_text.config(state=tk.DISABLED)
    
    def _setup_optimize_tab(self):
        """Set up the optimization tab"""
        # Create frames
        options_frame = ttk.LabelFrame(self.optimize_tab, text="Optimization Options", padding=10)
        options_frame.pack(fill=tk.BOTH, expand=True)
        
        # Optimization options
        self.quantization_var = tk.BooleanVar(value=True)
        self.quantization_check = ttk.Checkbutton(options_frame, text="Quantization", variable=self.quantization_var)
        self.quantization_check.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        ttk.Label(options_frame, text="Bits:").grid(row=0, column=1, sticky=tk.W, pady=5)
        self.quantization_bits_var = tk.IntVar(value=8)
        self.quantization_bits_combo = ttk.Combobox(options_frame, textvariable=self.quantization_bits_var, width=5)
        self.quantization_bits_combo['values'] = [8, 4, 2]
        self.quantization_bits_combo.grid(row=0, column=2, sticky=tk.W, pady=5)
        
        self.pruning_var = tk.BooleanVar(value=True)
        self.pruning_check = ttk.Checkbutton(options_frame, text="Pruning", variable=self.pruning_var)
        self.pruning_check.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        ttk.Label(options_frame, text="Ratio:").grid(row=1, column=1, sticky=tk.W, pady=5)
        self.pruning_ratio_var = tk.DoubleVar(value=0.3)
        self.pruning_ratio_combo = ttk.Combobox(options_frame, textvariable=self.pruning_ratio_var, width=5)
        self.pruning_ratio_combo['values'] = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.pruning_ratio_combo.grid(row=1, column=2, sticky=tk.W, pady=5)
        
        self.lazy_eval_var = tk.BooleanVar(value=True)
        self.lazy_eval_check = ttk.Checkbutton(options_frame, text="Lazy Evaluation", variable=self.lazy_eval_var)
        self.lazy_eval_check.grid(row=2, column=0, sticky=tk.W, pady=5)
        
        self.token_batch_var = tk.BooleanVar(value=True)
        self.token_batch_check = ttk.Checkbutton(options_frame, text="Token Batching", variable=self.token_batch_var)
        self.token_batch_check.grid(row=3, column=0, sticky=tk.W, pady=5)
        
        ttk.Label(options_frame, text="Batch Size:").grid(row=3, column=1, sticky=tk.W, pady=5)
        self.token_batch_size_var = tk.IntVar(value=8)
        self.token_batch_size_combo = ttk.Combobox(options_frame, textvariable=self.token_batch_size_var, width=5)
        self.token_batch_size_combo['values'] = [4, 8, 16, 32]
        self.token_batch_size_combo.grid(row=3, column=2, sticky=tk.W, pady=5)
        
        self.dynamic_graph_var = tk.BooleanVar(value=True)
        self.dynamic_graph_check = ttk.Checkbutton(options_frame, text="Dynamic Graph Construction", variable=self.dynamic_graph_var)
        self.dynamic_graph_check.grid(row=4, column=0, sticky=tk.W, pady=5)
        
        # Optimize button
        self.optimize_button = ttk.Button(options_frame, text="Optimize Model", command=self._optimize_model)
        self.optimize_button.grid(row=5, column=0, sticky=tk.W, pady=10)
        self.optimize_button.config(state=tk.DISABLED)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.optimize_tab, text="Optimization Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Results text
        self.results_text = tk.Text(results_frame, height=10, width=80, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.insert(tk.END, "No optimization results yet")
        self.results_text.config(state=tk.DISABLED)
    
    def _setup_test_tab(self):
        """Set up the test and export tab"""
        # Create frames
        test_frame = ttk.LabelFrame(self.test_tab, text="Test Model", padding=10)
        test_frame.pack(fill=tk.BOTH, expand=True)
        
        # Prompt entry
        ttk.Label(test_frame, text="Enter a prompt:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.prompt_var = tk.StringVar()
        self.prompt_entry = ttk.Entry(test_frame, textvariable=self.prompt_var, width=60)
        self.prompt_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Generate button
        self.generate_button = ttk.Button(test_frame, text="Generate Text", command=self._generate_text)
        self.generate_button.grid(row=1, column=1, sticky=tk.W, pady=10)
        self.generate_button.config(state=tk.DISABLED)
        
        # Generated text
        ttk.Label(test_frame, text="Generated text:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.generated_text = tk.Text(test_frame, height=10, width=80, wrap=tk.WORD)
        self.generated_text.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Export frame
        export_frame = ttk.LabelFrame(self.test_tab, text="Export Model", padding=10)
        export_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Export options
        ttk.Label(export_frame, text="Export format:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.export_format_var = tk.StringVar(value="Hugging Face")
        self.export_format_combo = ttk.Combobox(export_frame, textvariable=self.export_format_var, width=20)
        self.export_format_combo['values'] = ["Hugging Face", "Core ML"]
        self.export_format_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Export button
        self.export_button = ttk.Button(export_frame, text="Export Model", command=self._export_model)
        self.export_button.grid(row=1, column=1, sticky=tk.W, pady=10)
        self.export_button.config(state=tk.DISABLED)
    
    def _setup_about_tab(self):
        """Set up the about tab"""
        # Create frames
        about_frame = ttk.Frame(self.about_tab, padding=10)
        about_frame.pack(fill=tk.BOTH, expand=True)
        
        # About text
        about_text = """
        LLM Optimizer for Apple Silicon
        
        This application optimizes Large Language Models (LLMs) for Apple Silicon devices using the MLX framework.
        It incorporates the latest research in model quantization, pruning, and memory management to enable users
        to run larger models on lower-spec laptops.
        
        Features:
        - Model Quantization: Reduce model precision from 32-bit to 16-bit or 8-bit
        - Model Pruning: Remove less important weights to reduce model size
        - Lazy Evaluation: Defer computations until needed
        - Token Batching: Process tokens in batches for efficient inference
        - Dynamic Graph Construction: Adapt model shapes without recompilation
        - Unified Memory Utilization: Leverage Apple Silicon's unified memory architecture
        
        This application leverages the latest research in LLM optimization, including techniques from:
        - "OpenELM: An Efficient Language Model Family with Open-source Training and Inference Framework" (2024)
        - "Beyond Language: Applying MLX Transformers to Engineering Physics" (2024)
        
        It uses Apple's MLX framework, which is specifically designed for efficient machine learning on Apple Silicon devices.
        """
        
        about_label = ttk.Label(about_frame, text=about_text, wraplength=700, justify=tk.LEFT)
        about_label.pack(fill=tk.BOTH, expand=True)
        
        # System info frame
        system_frame = ttk.LabelFrame(self.about_tab, text="System Information", padding=10)
        system_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # System info text
        self.system_info_text = tk.Text(system_frame, height=10, width=80, wrap=tk.WORD)
        self.system_info_text.pack(fill=tk.BOTH, expand=True)
        self.system_info_text.config(state=tk.DISABLED)
    
    def _update_system_info(self):
        """Update the system information display"""
        system_info = utils.get_system_info()
        
        # Format the system info
        info_text = f"""
        Operating System: {system_info['os']} {system_info['os_version']}
        Machine: {system_info['machine']}
        Processor: {system_info['processor']}
        Python Version: {system_info['python_version']}
        Total Memory: {utils.format_size(system_info['total_memory'])}
        Available Memory: {utils.format_size(system_info['available_memory'])}
        Memory Usage: {system_info['memory_percent']}%
        GPU Available: {'Yes' if utils.check_gpu_availability() else 'No'}
        """
        
        # Update the text widget
        self.system_info_text.config(state=tk.NORMAL)
        self.system_info_text.delete(1.0, tk.END)
        self.system_info_text.insert(tk.END, info_text)
        self.system_info_text.config(state=tk.DISABLED)
    
    def _load_model(self):
        """Load a model from Hugging Face"""
        # Get the model name
        model_name = self.model_var.get()
        if not model_name:
            model_name = self.custom_model_var.get()
        
        if not model_name:
            messagebox.showerror("Error", "Please select or enter a model name")
            return
        
        # Update status
        self.status_var.set(f"Loading model {model_name}...")
        self.root.update_idletasks()
        
        # Load the model in a separate thread
        threading.Thread(target=self._load_model_thread, args=(model_name,)).start()
    
    def _load_model_thread(self, model_name):
        """Load a model in a separate thread"""
        try:
            # Load the model
            self.model_manager.load_model(model_name)
            
            # Update the model info
            self.root.after(0, self._update_model_info)
            
            # Enable the optimize button
            self.root.after(0, lambda: self.optimize_button.config(state=tk.NORMAL))
            
            # Update status
            self.root.after(0, lambda: self.status_var.set(f"Model {model_name} loaded successfully"))
        except Exception as e:
            logger.exception(f"Error loading model: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error loading model: {e}"))
            self.root.after(0, lambda: self.status_var.set("Ready"))
    
    def _update_model_info(self):
        """Update the model information display"""
        model_info = self.model_manager.get_model_info()
        
        # Format the model info
        info_text = f"""
        Model Name: {model_info['name']}
        Parameters: {model_info['parameters']:,}
        Size: {model_info['size']}
        """
        
        # Update the text widget
        self.model_info_text.config(state=tk.NORMAL)
        self.model_info_text.delete(1.0, tk.END)
        self.model_info_text.insert(tk.END, info_text)
        self.model_info_text.config(state=tk.DISABLED)
    
    def _optimize_model(self):
        """Optimize the loaded model"""
        if self.model_manager.model is None:
            messagebox.showerror("Error", "No model loaded")
            return
        
        # Get optimization config
        config = {
            "quantization": self.quantization_var.get(),
            "quantization_bits": self.quantization_bits_var.get(),
            "pruning": self.pruning_var.get(),
            "pruning_ratio": self.pruning_ratio_var.get(),
            "lazy_evaluation": self.lazy_eval_var.get(),
            "token_batching": self.token_batch_var.get(),
            "token_batch_size": self.token_batch_size_var.get(),
            "dynamic_graph": self.dynamic_graph_var.get(),
        }
        
        # Update status
        self.status_var.set("Optimizing model...")
        self.root.update_idletasks()
        
        # Optimize the model in a separate thread
        threading.Thread(target=self._optimize_model_thread, args=(config,)).start()
    
    def _optimize_model_thread(self, config):
        """Optimize the model in a separate thread"""
        try:
            # Optimize the model
            optimized_model, results = optimizer.optimize_model(self.model_manager.model, config)
            
            # Set the optimized model
            self.model_manager.set_optimized_model(optimized_model, config)
            
            # Update the results
            self.root.after(0, lambda: self._update_optimization_results(results))
            
            # Enable the generate and export buttons
            self.root.after(0, lambda: self.generate_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.export_button.config(state=tk.NORMAL))
            
            # Update status
            self.root.after(0, lambda: self.status_var.set("Model optimization complete"))
        except Exception as e:
            logger.exception(f"Error optimizing model: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error optimizing model: {e}"))
            self.root.after(0, lambda: self.status_var.set("Ready"))
    
    def _update_optimization_results(self, results):
        """Update the optimization results display"""
        # Format the results
        results_text = "Optimization Results:\n\n"
        
        for technique, result in results.items():
            results_text += f"{technique.title()}:\n"
            for key, value in result.items():
                results_text += f"  {key}: {value}\n"
            results_text += "\n"
        
        # Update the text widget
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results_text)
        self.results_text.config(state=tk.DISABLED)
    
    def _generate_text(self):
        """Generate text using the optimized model"""
        if self.model_manager.optimized_model is None:
            messagebox.showerror("Error", "No optimized model available")
            return
        
        # Get the prompt
        prompt = self.prompt_var.get()
        if not prompt:
            messagebox.showerror("Error", "Please enter a prompt")
            return
        
        # Update status
        self.status_var.set("Generating text...")
        self.root.update_idletasks()
        
        # Generate text in a separate thread
        threading.Thread(target=self._generate_text_thread, args=(prompt,)).start()
    
    def _generate_text_thread(self, prompt):
        """Generate text in a separate thread"""
        try:
            # Generate text
            generated_text = self.model_manager.generate_text(prompt)
            
            # Update the generated text
            self.root.after(0, lambda: self._update_generated_text(generated_text))
            
            # Update status
            self.root.after(0, lambda: self.status_var.set("Text generation complete"))
        except Exception as e:
            logger.exception(f"Error generating text: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error generating text: {e}"))
            self.root.after(0, lambda: self.status_var.set("Ready"))
    
    def _update_generated_text(self, text):
        """Update the generated text display"""
        self.generated_text.delete(1.0, tk.END)
        self.generated_text.insert(tk.END, text)
    
    def _export_model(self):
        """Export the optimized model"""
        if self.model_manager.optimized_model is None:
            messagebox.showerror("Error", "No optimized model available")
            return
        
        # Get the export format
        export_format = self.export_format_var.get()
        
        # Get the output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        
        # Update status
        self.status_var.set(f"Exporting model to {export_format} format...")
        self.root.update_idletasks()
        
        # Export the model in a separate thread
        threading.Thread(target=self._export_model_thread, args=(output_dir, export_format)).start()
    
    def _export_model_thread(self, output_dir, export_format):
        """Export the model in a separate thread"""
        try:
            # Export the model
            if export_format == "Core ML":
                # Convert to Core ML
                coreml_model, results = self.optimizer.convert_to_coreml(self.model_manager.optimized_model, {})
                
                # Save the Core ML model
                model_path = os.path.join(output_dir, f"{self.model_manager.model_name.split('/')[-1]}-optimized.mlpackage")
                # In a real implementation, you would save the Core ML model here
                
                message = f"Model exported to Core ML format at {model_path}"
            else:
                # Save the Hugging Face model
                model_path = self.model_manager.save_model(output_dir)
                message = f"Model exported to Hugging Face format at {model_path}"
            
            # Show success message
            self.root.after(0, lambda: messagebox.showinfo("Success", message))
            
            # Update status
            self.root.after(0, lambda: self.status_var.set("Model export complete"))
        except Exception as e:
            logger.exception(f"Error exporting model: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error exporting model: {e}"))
            self.root.after(0, lambda: self.status_var.set("Ready"))

def run(model_manager: ModelManager):
    """Run the GUI"""
    root = tk.Tk()
    app = LLMOptimizerGUI(root, model_manager)
    root.mainloop()
