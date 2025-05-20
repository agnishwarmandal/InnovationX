"""
GUI for the MQTM system.
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

class MQTMGUI:
    """GUI for the MQTM system."""
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("MQTM System")
        self.root.geometry("800x600")
        
        # Create notebook
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.train_tab = ttk.Frame(self.notebook)
        self.generate_tab = ttk.Frame(self.notebook)
        self.visualize_tab = ttk.Frame(self.notebook)
        self.trade_tab = ttk.Frame(self.notebook)
        self.pipeline_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text="Data")
        self.notebook.add(self.train_tab, text="Train")
        self.notebook.add(self.generate_tab, text="Generate")
        self.notebook.add(self.visualize_tab, text="Visualize")
        self.notebook.add(self.trade_tab, text="Trade")
        self.notebook.add(self.pipeline_tab, text="Pipeline")
        
        # Create console output
        self.console_frame = ttk.LabelFrame(root, text="Console Output")
        self.console_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.console = tk.Text(self.console_frame, wrap=tk.WORD, bg="black", fg="white")
        self.console.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize tabs
        self.init_data_tab()
        self.init_train_tab()
        self.init_generate_tab()
        self.init_visualize_tab()
        self.init_trade_tab()
        self.init_pipeline_tab()
        
        # Process queue
        self.process = None
    
    def init_data_tab(self):
        """Initialize the data tab."""
        # Create frame for inputs
        input_frame = ttk.Frame(self.data_tab)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create symbols input
        ttk.Label(input_frame, text="Symbols:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.data_symbols_var = tk.StringVar(value="BTCUSDT ETHUSDT")
        ttk.Entry(input_frame, textvariable=self.data_symbols_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create days input
        ttk.Label(input_frame, text="Days:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.data_days_var = tk.StringVar(value="30")
        ttk.Entry(input_frame, textvariable=self.data_days_var).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create button frame
        button_frame = ttk.Frame(self.data_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create download button
        ttk.Button(button_frame, text="Download Data", command=self.download_data).pack(side=tk.LEFT, padx=5, pady=5)
    
    def init_train_tab(self):
        """Initialize the train tab."""
        # Create frame for inputs
        input_frame = ttk.Frame(self.train_tab)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create symbols input
        ttk.Label(input_frame, text="Symbols:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.train_symbols_var = tk.StringVar(value="BTCUSDT ETHUSDT")
        ttk.Entry(input_frame, textvariable=self.train_symbols_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create batch size input
        ttk.Label(input_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.train_batch_size_var = tk.StringVar(value="32")
        ttk.Entry(input_frame, textvariable=self.train_batch_size_var).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create epochs input
        ttk.Label(input_frame, text="Epochs:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.train_epochs_var = tk.StringVar(value="50")
        ttk.Entry(input_frame, textvariable=self.train_epochs_var).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create learning rate input
        ttk.Label(input_frame, text="Learning Rate:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.train_lr_var = tk.StringVar(value="0.0001")
        ttk.Entry(input_frame, textvariable=self.train_lr_var).grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create output directory input
        ttk.Label(input_frame, text="Output Directory:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.train_output_dir_var = tk.StringVar(value="models")
        ttk.Entry(input_frame, textvariable=self.train_output_dir_var).grid(row=4, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=lambda: self.browse_directory(self.train_output_dir_var)).grid(row=4, column=2, padx=5, pady=5)
        
        # Create button frame
        button_frame = ttk.Frame(self.train_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create train button
        ttk.Button(button_frame, text="Train Models", command=self.train_models).pack(side=tk.LEFT, padx=5, pady=5)
    
    def init_generate_tab(self):
        """Initialize the generate tab."""
        # Create frame for inputs
        input_frame = ttk.Frame(self.generate_tab)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create num samples input
        ttk.Label(input_frame, text="Number of Samples:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.generate_num_samples_var = tk.StringVar(value="1000")
        ttk.Entry(input_frame, textvariable=self.generate_num_samples_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create sigma multiplier input
        ttk.Label(input_frame, text="Sigma Multiplier:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.generate_sigma_var = tk.StringVar(value="1.5")
        ttk.Entry(input_frame, textvariable=self.generate_sigma_var).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create output directory input
        ttk.Label(input_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.generate_output_dir_var = tk.StringVar(value="samples")
        ttk.Entry(input_frame, textvariable=self.generate_output_dir_var).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=lambda: self.browse_directory(self.generate_output_dir_var)).grid(row=2, column=2, padx=5, pady=5)
        
        # Create button frame
        button_frame = ttk.Frame(self.generate_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create generate button
        ttk.Button(button_frame, text="Generate Samples", command=self.generate_samples).pack(side=tk.LEFT, padx=5, pady=5)
    
    def init_visualize_tab(self):
        """Initialize the visualize tab."""
        # Create frame for inputs
        input_frame = ttk.Frame(self.visualize_tab)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create component input
        ttk.Label(input_frame, text="Component:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.visualize_component_var = tk.StringVar(value="tqe")
        ttk.Combobox(input_frame, textvariable=self.visualize_component_var, values=["tqe", "sp3", "mg"]).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create symbol input
        ttk.Label(input_frame, text="Symbol:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.visualize_symbol_var = tk.StringVar(value="BTCUSDT")
        ttk.Entry(input_frame, textvariable=self.visualize_symbol_var).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create output directory input
        ttk.Label(input_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.visualize_output_dir_var = tk.StringVar(value="visualizations")
        ttk.Entry(input_frame, textvariable=self.visualize_output_dir_var).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=lambda: self.browse_directory(self.visualize_output_dir_var)).grid(row=2, column=2, padx=5, pady=5)
        
        # Create button frame
        button_frame = ttk.Frame(self.visualize_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create visualize button
        ttk.Button(button_frame, text="Visualize", command=self.visualize).pack(side=tk.LEFT, padx=5, pady=5)
    
    def init_trade_tab(self):
        """Initialize the trade tab."""
        # Create frame for inputs
        input_frame = ttk.Frame(self.trade_tab)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create symbol input
        ttk.Label(input_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.trade_symbol_var = tk.StringVar(value="BTCUSDT")
        ttk.Entry(input_frame, textvariable=self.trade_symbol_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create mode input
        ttk.Label(input_frame, text="Mode:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.trade_mode_var = tk.StringVar(value="backtest")
        ttk.Combobox(input_frame, textvariable=self.trade_mode_var, values=["backtest", "paper", "live"]).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create days input
        ttk.Label(input_frame, text="Days:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.trade_days_var = tk.StringVar(value="7")
        ttk.Entry(input_frame, textvariable=self.trade_days_var).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create capital input
        ttk.Label(input_frame, text="Capital:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.trade_capital_var = tk.StringVar(value="10000.0")
        ttk.Entry(input_frame, textvariable=self.trade_capital_var).grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create risk per trade input
        ttk.Label(input_frame, text="Risk Per Trade:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.trade_risk_var = tk.StringVar(value="0.01")
        ttk.Entry(input_frame, textvariable=self.trade_risk_var).grid(row=4, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create button frame
        button_frame = ttk.Frame(self.trade_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create trade button
        ttk.Button(button_frame, text="Run Trading", command=self.run_trading).pack(side=tk.LEFT, padx=5, pady=5)
    
    def init_pipeline_tab(self):
        """Initialize the pipeline tab."""
        # Create frame for inputs
        input_frame = ttk.Frame(self.pipeline_tab)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create symbols input
        ttk.Label(input_frame, text="Symbols:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.pipeline_symbols_var = tk.StringVar(value="BTCUSDT ETHUSDT")
        ttk.Entry(input_frame, textvariable=self.pipeline_symbols_var).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create download data checkbox
        self.pipeline_download_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(input_frame, text="Download Data", variable=self.pipeline_download_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Create days input
        ttk.Label(input_frame, text="Days:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.pipeline_days_var = tk.StringVar(value="30")
        ttk.Entry(input_frame, textvariable=self.pipeline_days_var).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create batch size input
        ttk.Label(input_frame, text="Batch Size:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.pipeline_batch_size_var = tk.StringVar(value="32")
        ttk.Entry(input_frame, textvariable=self.pipeline_batch_size_var).grid(row=3, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create epochs input
        ttk.Label(input_frame, text="Epochs:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.pipeline_epochs_var = tk.StringVar(value="10")
        ttk.Entry(input_frame, textvariable=self.pipeline_epochs_var).grid(row=4, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Create output directory input
        ttk.Label(input_frame, text="Output Directory:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.pipeline_output_dir_var = tk.StringVar(value="pipeline_output")
        ttk.Entry(input_frame, textvariable=self.pipeline_output_dir_var).grid(row=5, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=lambda: self.browse_directory(self.pipeline_output_dir_var)).grid(row=5, column=2, padx=5, pady=5)
        
        # Create button frame
        button_frame = ttk.Frame(self.pipeline_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create pipeline button
        ttk.Button(button_frame, text="Run Pipeline", command=self.run_pipeline).pack(side=tk.LEFT, padx=5, pady=5)
    
    def browse_directory(self, var):
        """Browse for a directory."""
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)
    
    def download_data(self):
        """Download data."""
        # Build command
        cmd = [
            "python", "mqtm_cli.py",
            "data",
            "--symbols"
        ] + self.data_symbols_var.get().split() + [
            "--days", self.data_days_var.get()
        ]
        
        # Run command
        self.run_command(cmd)
    
    def train_models(self):
        """Train models."""
        # Build command
        cmd = [
            "python", "mqtm_cli.py",
            "train",
            "--symbols"
        ] + self.train_symbols_var.get().split() + [
            "--batch_size", self.train_batch_size_var.get(),
            "--epochs", self.train_epochs_var.get(),
            "--learning_rate", self.train_lr_var.get(),
            "--output_dir", self.train_output_dir_var.get()
        ]
        
        # Run command
        self.run_command(cmd)
    
    def generate_samples(self):
        """Generate samples."""
        # Build command
        cmd = [
            "python", "mqtm_cli.py",
            "generate",
            "--num_samples", self.generate_num_samples_var.get(),
            "--sigma_multiplier", self.generate_sigma_var.get(),
            "--output_dir", self.generate_output_dir_var.get()
        ]
        
        # Run command
        self.run_command(cmd)
    
    def visualize(self):
        """Visualize components."""
        # Build command
        cmd = [
            "python", "mqtm_cli.py",
            "visualize",
            "--component", self.visualize_component_var.get(),
            "--symbol", self.visualize_symbol_var.get(),
            "--output_dir", self.visualize_output_dir_var.get()
        ]
        
        # Run command
        self.run_command(cmd)
    
    def run_trading(self):
        """Run trading."""
        # Build command
        cmd = [
            "python", "mqtm_cli.py",
            "trade",
            "--symbol", self.trade_symbol_var.get(),
            "--mode", self.trade_mode_var.get(),
            "--days", self.trade_days_var.get(),
            "--capital", self.trade_capital_var.get(),
            "--risk_per_trade", self.trade_risk_var.get()
        ]
        
        # Run command
        self.run_command(cmd)
    
    def run_pipeline(self):
        """Run pipeline."""
        # Build command
        cmd = [
            "python", "mqtm_cli.py",
            "pipeline",
            "--symbols"
        ] + self.pipeline_symbols_var.get().split() + [
            "--batch_size", self.pipeline_batch_size_var.get(),
            "--epochs", self.pipeline_epochs_var.get(),
            "--output_dir", self.pipeline_output_dir_var.get()
        ]
        
        if self.pipeline_download_var.get():
            cmd.append("--download_data")
            cmd.extend(["--days", self.pipeline_days_var.get()])
        
        # Run command
        self.run_command(cmd)
    
    def run_command(self, cmd):
        """Run a command in a separate thread."""
        # Check if a process is already running
        if self.process is not None and self.process.poll() is None:
            messagebox.showerror("Error", "A process is already running.")
            return
        
        # Clear console
        self.console.delete(1.0, tk.END)
        
        # Update status
        self.status_var.set(f"Running: {' '.join(cmd)}")
        
        # Start process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Start thread to read output
        threading.Thread(target=self.read_output, daemon=True).start()
    
    def read_output(self):
        """Read output from process."""
        for line in iter(self.process.stdout.readline, ""):
            self.console.insert(tk.END, line)
            self.console.see(tk.END)
        
        # Process finished
        self.process.stdout.close()
        return_code = self.process.wait()
        
        # Update status
        if return_code == 0:
            self.status_var.set("Command completed successfully.")
        else:
            self.status_var.set(f"Command failed with return code {return_code}.")

def main():
    """Main function."""
    root = tk.Tk()
    app = MQTMGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
