import os
import csv
import time
import pandas as pd
import torch
from typing import Dict, List, Tuple

class ModelPerformanceTracker:
    """
    Tracks and records model performance metrics to CSV files
    """
    def __init__(self, output_path: str = None):
        """
        Initialize the performance tracker
        
        Args:
            output_path: Path to save the CSV files. If None, saves to Results/metrics/
        """
        self.metrics = []
        self.model_timings = {}
        self.model_sizes = {}
        
        # Create default output directory if not provided
        if output_path is None:
            self.output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                           'Results', 'metrics')
        else:
            self.output_path = output_path
            
        # Ensure directory exists
        os.makedirs(self.output_path, exist_ok=True)
        
    def start_timing(self, model_name: str):
        """Start timing a model's execution"""
        self.model_timings[model_name] = time.time()
    
    def end_timing(self, model_name: str) -> float:
        """End timing a model's execution and return duration in seconds"""
        if model_name in self.model_timings:
            duration = time.time() - self.model_timings[model_name]
            return duration
        return 0.0
    
    def get_model_size(self, model) -> int:
        """Get the number of parameters in a model"""
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        return sum([torch.numel(p) for p in model_parameters])
    
    def record_metrics(self, model_name: str, snr: float, final_ser: float, 
                       model_size: int, run_time: float):
        """Record metrics for a model"""
        self.metrics.append({
            'model': model_name,
            'snr': snr,
            'final-ser': final_ser,
            'model-size': model_size,
            'model-run-time': run_time
        })
    
    def save_to_csv(self, filename: str = None):
        """Save recorded metrics to CSV"""
        if not self.metrics:
            print("No metrics to save")
            return
            
        if filename is None:
            filename = f"model_performance_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            
        file_path = os.path.join(self.output_path, filename)
        
        df = pd.DataFrame(self.metrics)
        df.to_csv(file_path, index=False)
        print(f"Metrics saved to {file_path}")
        
        # Also print a summary
        print("\nModel Performance Summary:")
        print("="*60)
        print(df)
        print("="*60)
        
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Return metrics as a DataFrame"""
        if not self.metrics:
            return pd.DataFrame()
        return pd.DataFrame(self.metrics)
