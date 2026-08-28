import os
import csv
import time
import math
import torch
from typing import Dict, List

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
        self._write_rows(file_path, self.metrics)
        print(f"Metrics saved to {file_path}")
        
        # Also print a summary
        print("\nModel Performance Summary:")
        print("="*60)
        self._print_table(self.metrics)
        print("="*60)
        
    @staticmethod
    def _write_rows(file_path: str, rows: List[Dict]):
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with open(file_path, 'w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _print_table(rows: List[Dict]):
        if not rows:
            return
        columns = list(dict.fromkeys(key for row in rows for key in row))
        print('\t'.join(columns))
        for row in rows:
            print('\t'.join(str(row.get(column, '')) for column in columns))

    def get_summary_dataframe(self) -> List[Dict]:
        """Return recorded metrics as a list of table rows."""
        return list(self.metrics)

    def save_ber_vs_snr_table(self, filename: str = None) -> List[Dict]:
        """Pivot results into the model-vs-SNR table used in the paper.

        Rows are SNR, columns are models, values are mean SER - matching the
        benchmark table layout in the README.
        """
        rows = self.get_summary_dataframe()
        if not rows:
            print("No metrics to pivot")
            return rows

        value_col = 'ser-mean' if any('ser-mean' in row for row in rows) else 'final-ser'
        config_cols = [
            column for column in (
                'memory_length', 'noisy_est_var', 'train_samples',
                'channel_coefficients', 'n_symbols', 'val_block_length',
                'pilots_num', 'detector_method'
            ) if any(column in row for row in rows)
        ]
        grouped = {}
        for row in rows:
            key = tuple(row.get(column) for column in ['snr'] + config_cols)
            model = row['model']
            grouped.setdefault((key, model), []).append(row.get(value_col, 0))
        models = sorted({row['model'] for row in rows})
        table = []
        for key in sorted({key for key, _ in grouped}, key=lambda value: tuple(str(v) for v in value)):
            output = dict(zip(['snr'] + config_cols, key))
            for model in models:
                values = grouped.get((key, model), [])
                if values:
                    output[model] = sum(values) / len(values)
            table.append(output)

        if filename is None:
            filename = f"ber_vs_snr_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(self.output_path, filename)
        self._write_rows(file_path, table)
        print(f"BER-vs-SNR table saved to {file_path}")
        self._print_table(table)
        return table

    def save_complexity_table(self, filename: str = None) -> List[Dict]:
        """Export the BER-gain-vs-complexity comparison the reviewers requested."""
        rows = self.get_summary_dataframe()
        if not rows:
            print("No metrics to summarise")
            return rows

        cols = sorted({c for row in rows for c in row if c.startswith('complexity-')})
        if not cols:
            print("No complexity metrics recorded")
            return []

        value_col = 'ser-mean' if any('ser-mean' in row for row in rows) else 'final-ser'
        config_cols = [
            column for column in (
                'snr', 'memory_length', 'noisy_est_var', 'train_samples',
                'channel_coefficients', 'n_symbols', 'val_block_length',
                'pilots_num', 'detector_method'
            ) if any(column in row for row in rows)
        ]
        groups = {}
        for row in rows:
            key = tuple(row.get(column) for column in ['model'] + config_cols)
            groups.setdefault(key, []).append(row)
        summary = []
        for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
            output = dict(zip(['model'] + config_cols, key))
            for column in cols:
                output[column] = sum(row.get(column, 0) for row in group) / len(group)
            output[value_col] = sum(row.get(value_col, 0) for row in group) / len(group)
            output['model-size'] = group[0].get('model-size')
            summary.append(output)

        if filename is None:
            filename = f"complexity_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(self.output_path, filename)
        self._write_rows(file_path, summary)
        print(f"Complexity table saved to {file_path}")
        self._print_table(summary)
        return summary
