import os
import csv
import time
import math
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
                       model_size: int, run_time: float,
                       config: Dict = None, complexity: Dict = None,
                       ser_reps: List[float] = None):
        """Record metrics for a model.

        :param config: experiment settings (memory_length, noisy_est_var, train
            samples, ...). Recording these is what makes rows from different
            sweeps distinguishable after the fact.
        :param complexity: output of Code.complexity.profile_model.
        :param ser_reps: per-repetition SER values, used to derive a confidence
            interval. Reporting a mean without a CI makes it impossible to tell
            whether a gap between two models is real.
        """
        row = {
            'model': model_name,
            'snr': snr,
            'final-ser': final_ser,
            'model-size': model_size,
            'model-run-time': run_time
        }

        if ser_reps is not None and len(ser_reps) > 0:
            row.update(self._confidence_interval(ser_reps))

        if config:
            for key in ('memory_length', 'noisy_est_var', 'train_samples',
                        'channel_coefficients', 'n_symbols', 'val_block_length',
                        'detector_method'):
                if key in config:
                    row[key] = config[key]

        if complexity:
            for key, value in complexity.items():
                row[f'complexity-{key}'] = value

        self.metrics.append(row)

    @staticmethod
    def _confidence_interval(values: List[float], confidence_z: float = 1.96) -> Dict:
        """Mean and normal-approximation CI half-width over repetitions."""
        n = len(values)
        mean = sum(values) / n
        if n < 2:
            return {'ser-mean': mean, 'ser-std': 0.0, 'ser-ci95': 0.0, 'ser-reps': n}
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)
        # standard error of the mean
        half_width = confidence_z * std / math.sqrt(n)
        return {'ser-mean': mean, 'ser-std': std, 'ser-ci95': half_width, 'ser-reps': n}
    
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

    def save_ber_vs_snr_table(self, filename: str = None) -> pd.DataFrame:
        """Pivot results into the model-vs-SNR table used in the paper.

        Rows are SNR, columns are models, values are mean SER - matching the
        benchmark table layout in the README.
        """
        df = self.get_summary_dataframe()
        if df.empty:
            print("No metrics to pivot")
            return df

        value_col = 'ser-mean' if 'ser-mean' in df.columns else 'final-ser'
        table = df.pivot_table(index='snr', columns='model',
                               values=value_col, aggfunc='mean')

        if filename is None:
            filename = f"ber_vs_snr_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(self.output_path, filename)
        table.to_csv(file_path)
        print(f"BER-vs-SNR table saved to {file_path}")
        print(table)
        return table

    def save_complexity_table(self, filename: str = None) -> pd.DataFrame:
        """Export the BER-gain-vs-complexity comparison the reviewers requested."""
        df = self.get_summary_dataframe()
        if df.empty:
            print("No metrics to summarise")
            return df

        cols = [c for c in df.columns if c.startswith('complexity-')]
        if not cols:
            print("No complexity metrics recorded")
            return pd.DataFrame()

        value_col = 'ser-mean' if 'ser-mean' in df.columns else 'final-ser'
        agg = {c: 'first' for c in cols}
        agg[value_col] = 'mean'
        agg['model-size'] = 'first'
        summary = df.groupby('model').agg(agg).reset_index()

        if filename is None:
            filename = f"complexity_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(self.output_path, filename)
        summary.to_csv(file_path, index=False)
        print(f"Complexity table saved to {file_path}")
        print(summary)
        return summary
