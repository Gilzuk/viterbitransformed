"""
Data caching system for pre-generating and loading channel data
This significantly reduces CPU usage by generating data once and loading from disk
"""
import numpy as np
import torch
import os
import pickle
import math
from typing import Tuple, Dict
from pathlib import Path


class ChannelDataCache:
    """Manages pre-generated channel data to reduce CPU overhead"""
    
    def __init__(self, cache_dir: str = "Data_Cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        print(f"[DataCache] Cache directory: {self.cache_dir.absolute()}")
    
    def get_cache_filename(self, snr: float, gamma: float, block_length: int,
                          transmission_length: int, words: int,
                          channel_coefficients: str, phase: str,
                          memory_length: int, noisy_est_var: float,
                          fading_taps_type: int, n_symbols: int,
                          fading: bool, rep: int = None) -> str:
        """Generate unique cache filename based on parameters.

        Every parameter that changes the generated data must appear here, otherwise
        data generated under one setting would be silently reused under another.

        `rep` distinguishes independent repeated draws under otherwise-identical
        parameters (e.g. Monte-Carlo evaluation repetitions) -- omitted (None) by
        any caller that wants the old "one cached draw, reused" behavior (e.g. a
        single fixed training/validation set reused across minibatches/models).
        """
        rep_suffix = f'_rep{rep}' if rep is not None else ''
        filename = (f"data_snr{snr}_gamma{gamma}_bl{block_length}_tl{transmission_length}"
                    f"_w{words}_ch{channel_coefficients}_ml{memory_length}_nev{noisy_est_var}"
                    f"_ftt{fading_taps_type}_ns{n_symbols}_fad{int(fading)}{rep_suffix}_{phase}.pkl")
        return str(self.cache_dir / filename)
    
    def cache_exists(self, filename: str) -> bool:
        """Check if cache file exists"""
        return os.path.exists(filename)
    
    def save_to_cache(self, filename: str, b: np.ndarray, y: np.ndarray,
                     snr: float, gamma: float, block_length: int,
                     transmission_length: int, words: int,
                     channel_coefficients: str, phase: str,
                     memory_length: int, noisy_est_var: float,
                     fading_taps_type: int, n_symbols: int, fading: bool,
                     rep: int = None):
        """Save generated data to cache file with metadata for validation"""
        print(f"[DataCache] Saving to cache: {Path(filename).name}")
        metadata = {
            'snr': snr,
            'gamma': gamma,
            'block_length': block_length,
            'transmission_length': transmission_length,
            'words': words,
            'channel_coefficients': channel_coefficients,
            'phase': phase,
            'memory_length': memory_length,
            'noisy_est_var': noisy_est_var,
            'fading_taps_type': fading_taps_type,
            'n_symbols': n_symbols,
            'fading': fading,
            'rep': rep
        }
        with open(filename, 'wb') as f:
            pickle.dump({'b': b, 'y': y, 'metadata': metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"[DataCache] Saved {file_size_mb:.2f} MB to disk")
    
    def validate_cache(self, filename: str, snr: float, gamma: float,
                      block_length: int, transmission_length: int,
                      words: int, channel_coefficients: str, phase: str,
                      memory_length: int, noisy_est_var: float,
                      fading_taps_type: int, n_symbols: int,
                      fading: bool, rep: int = None) -> bool:
        """Check if cached file matches expected parameters"""
        if not os.path.exists(filename):
            return False
        
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            # Check if metadata exists (old cache files won't have it)
            if 'metadata' not in data:
                print(f"[DataCache] Cache file missing metadata, will regenerate: {Path(filename).name}")
                return False
            
            metadata = data['metadata']

            expected = {
                'snr': snr,
                'gamma': gamma,
                'block_length': block_length,
                'transmission_length': transmission_length,
                'words': words,
                'channel_coefficients': channel_coefficients,
                'phase': phase,
                'memory_length': memory_length,
                'noisy_est_var': noisy_est_var,
                'fading_taps_type': fading_taps_type,
                'n_symbols': n_symbols,
                'fading': fading,
                'rep': rep
            }

            # Cache files written before these keys existed cannot be trusted
            missing = [k for k in expected if k not in metadata]
            if missing:
                print(f"[DataCache] Cache file missing keys {missing}, will regenerate: {Path(filename).name}")
                return False

            float_keys = {'snr', 'gamma', 'noisy_est_var'}
            mismatched = {
                k: (v, metadata[k]) for k, v in expected.items()
                if (
                    not math.isclose(float(metadata[k]), float(v), rel_tol=1e-9, abs_tol=1e-12)
                    if k in float_keys
                    else metadata[k] != v
                )
            }
            if mismatched:
                print(f"[DataCache] Parameter mismatch, will regenerate: {Path(filename).name}")
                for key, (want, got) in mismatched.items():
                    print(f"  {key}: expected={want}, found={got}")
                return False
            
            return True
        except Exception as e:
            print(f"[DataCache] Error validating cache file {Path(filename).name}: {e}")
            return False
    
    def load_from_cache(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from cache file"""
        print(f"[DataCache] Loading from cache: {Path(filename).name}")
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data['b'], data['y']
    
    def load_to_gpu_chunks(self, filename: str, device: torch.device, 
                          chunk_size: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load data from cache directly to GPU with dynamic chunk sizing based on available memory"""
        
        # Load from disk first
        b, y = self.load_from_cache(filename)
        
        # Determine optimal chunk size based on available GPU memory
        if device.type == "cuda" and chunk_size is None:
            gpu_memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
            total_samples = b.shape[0]
            
            # Estimate memory needed per sample (very rough estimate)
            # Assuming ~10MB per sample for sequences of length ~1200
            estimated_memory_per_sample_gb = 0.01
            max_safe_samples = int((gpu_memory_free * 0.7) / estimated_memory_per_sample_gb)  # Use 70% of free memory
            
            chunk_size = min(max(50, max_safe_samples), total_samples)
            print(f"[DataCache] GPU Memory: {gpu_memory_free:.2f} GB free, loading {total_samples} samples in chunks of {chunk_size}")
        elif chunk_size is None:
            chunk_size = b.shape[0]  # CPU - load all at once
        else:
            print(f"[DataCache] Loading to GPU in chunks of {chunk_size}")
        
        # Convert to tensors on GPU in chunks
        b_chunks = []
        y_chunks = []
        
        for i in range(0, len(b), chunk_size):
            end_idx = min(i + chunk_size, len(b))
            b_chunk = torch.tensor(b[i:end_idx], dtype=torch.float32, device=device)
            y_chunk = torch.tensor(y[i:end_idx], dtype=torch.float32, device=device)
            b_chunks.append(b_chunk)
            y_chunks.append(y_chunk)
            
            if len(b) > chunk_size:
                print(f"[DataCache] Loaded chunk {i//chunk_size + 1}/{(len(b)-1)//chunk_size + 1}")
        
        # Concatenate all chunks
        b_gpu = torch.cat(b_chunks, dim=0) if len(b_chunks) > 1 else b_chunks[0]
        y_gpu = torch.cat(y_chunks, dim=0) if len(y_chunks) > 1 else y_chunks[0]
        
        del b_chunks, y_chunks  # Free intermediate memory
        
        print(f"[DataCache] Loaded to GPU: {b_gpu.shape}, {y_gpu.shape}")
        return b_gpu, y_gpu
    
    def clear_cache(self):
        """Remove all cached files"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            print(f"[DataCache] Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cached data"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            'num_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'files': [f.name for f in cache_files]
        }
