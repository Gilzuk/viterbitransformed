from Code.dir_definitions import *
from Code.plotter import get_ser_data, plot_ser_by_block_index, plot_ser_by_snr, plot_summary_table
from Code.trainer import Trainer
from Code.csv_reporter import ModelPerformanceTracker
from Code.complexity import profile_model
import torch
import gc
import numpy as np
import time
from datetime import datetime
import psutil
import threading
import sys
import logging

# Import data cache for pre-generation
from Code.channel.data_cache import ChannelDataCache

# ============================================================================
# Logging Setup - Dual output to console and log file
# ============================================================================

class TeeLogger:
    """Captures stdout/stderr to both console and log file"""
    def __init__(self, log_file, stream):
        self.log_file = log_file
        self.stream = stream
        
    def write(self, message):
        self.stream.write(message)
        self.stream.flush()
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.stream.flush()
        self.log_file.flush()

# Create logs directory if it doesn't exist
log_dir = os.path.join(ROOT_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Create log filename with timestamp
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f'training_log_{log_timestamp}.log')

# Open log file and redirect stdout/stderr
log_file = open(log_filename, 'w', encoding='utf-8')
sys.stdout = TeeLogger(log_file, sys.__stdout__)
sys.stderr = TeeLogger(log_file, sys.__stderr__)

print(f"{'='*70}")
print(f"📝 LOGGING ENABLED")
print(f"Log file: {log_filename}")
print(f"All output will be saved to log file and displayed in terminal")
print(f"{'='*70}\n")

# Device selection and CUDA/CPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU Memory Cleanup - Reset and clear memory from previous runs
if device.type == "cuda":
    print(f"{'='*60}")
    print(f"🧹 CLEANING GPU MEMORY FROM PREVIOUS RUNS...")
    print(f"{'='*60}")
    
    # Clear cached memory
    torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Reset CUDA memory allocator stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    # Synchronize to ensure all operations are complete
    torch.cuda.synchronize()
    
    print(f"✅ GPU memory cleaned and reset")
    print(f"{'='*60}\n")

# Print GPU information
if device.type == "cuda":
    print(f"{'='*60}")
    print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}")
    
    # Show memory state after cleanup
    mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
    mem_free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - mem_allocated
    print(f"GPU Memory After Cleanup:")
    print(f"  Allocated: {mem_allocated:.3f} GB")
    print(f"  Reserved:  {mem_reserved:.3f} GB")
    print(f"  Free:      {mem_free:.3f} GB")
    print(f"{'='*60}")
else:
    print(f"{'='*60}")
    print(f"RUNNING ON CPU (No CUDA GPU available)")
    print(f"{'='*60}")

# No mixed precision - using standard FP32
use_amp = False
dtype = torch.float32
scaler = None

print(f"PRECISION: Standard Training (FP32)")
print(f"{'='*60}\n")

# Enable cuDNN optimization
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Limit CPU threads to reduce CPU load (critical for reducing CPU usage)
# Reduce to 2 threads for PyTorch CPU operations
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
print(f"CPU threads limited to 2 for reduced CPU usage")

# Enable memory fragmentation management to avoid OOM errors
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Global monitoring variables
monitoring_active = False
monitor_thread = None

# Verbosity control - set to False to silence pipeline prints
VERBOSE = False  # Set to False for silent mode

def print_utilization():
    """Print current GPU and CPU utilization"""
    if device.type == "cuda":
        # GPU metrics
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_memory_free = gpu_memory_total - gpu_memory_allocated
        gpu_util_percent = (gpu_memory_allocated / gpu_memory_total) * 100
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        memory = psutil.virtual_memory()
        
        print(f"\n{'='*70}")
        print(f"GPU UTILIZATION:")
        print(f"  Memory Used:     {gpu_memory_allocated:.2f} GB / {gpu_memory_total:.2f} GB ({gpu_util_percent:.1f}%)")
        print(f"  Memory Free:     {gpu_memory_free:.2f} GB")
        print(f"  Memory Reserved: {gpu_memory_reserved:.2f} GB")
        print(f"CPU UTILIZATION:")
        print(f"  CPU Usage (avg): {cpu_percent:.1f}%")
        print(f"  Per-core usage:  {', '.join([f'{c:.0f}%' for c in cpu_per_core])}")
        print(f"  RAM Usage:       {memory.percent:.1f}% ({memory.used / 1024**3:.2f} GB / {memory.total / 1024**3:.2f} GB)")
        print(f"  Active Threads:  {threading.active_count()}")
        print(f"{'='*70}\n")
    else:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        print(f"\n{'='*70}")
        print(f"CPU UTILIZATION:")
        print(f"  CPU Usage: {cpu_percent:.1f}%")
        print(f"  RAM Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.2f} GB / {memory.total / 1024**3:.2f} GB)")
        print(f"{'='*70}\n")

def monitor_utilization_background(interval=30):
    """Background thread to monitor utilization periodically"""
    global monitoring_active
    while monitoring_active:
        time.sleep(interval)
        if monitoring_active:
            print_utilization()

def start_monitoring(interval=30):
    """Start background monitoring thread"""
    global monitoring_active, monitor_thread
    monitoring_active = True
    monitor_thread = threading.Thread(target=monitor_utilization_background, args=(interval,), daemon=True)
    monitor_thread.start()
    print(f"📊 Background monitoring started (updates every {interval}s)")

def stop_monitoring():
    """Stop background monitoring thread"""
    global monitoring_active
    monitoring_active = False
    print("📊 Background monitoring stopped")

def pre_generate_data(parameters, gamma, hyperparams):
    """Pre-generate and cache data for all parameter combinations to reduce CPU load during training
    
    Args:
        parameters: List of (snr, val_block_length, pilot_length) tuples
        gamma: Gamma value for channel
        hyperparams: Dictionary of hyperparameters
    """
    if VERBOSE:
        print("\n" + "="*70)
        print("PRE-GENERATING DATA FOR ALL PARAMETER COMBINATIONS")
        print("This will take a few minutes but will drastically reduce CPU usage later")
        print("="*70 + "\n")
    
    cache = ChannelDataCache()
    
    # Get cache stats before
    stats_before = cache.get_cache_stats()
    if VERBOSE:
        print(f"📦 Cache before: {stats_before['num_files']} files, {stats_before['total_size_mb']:.2f} MB\n")
    
    # Import here to avoid circular dependency
    from Code.channel.channel_dataset import ChannelModelDataset
    from numpy.random import RandomState
    
    total_datasets = len(parameters) * 2  # train + val for each parameter set
    current_dataset = 0
    
    for snr, val_block_length, pilot_length in parameters:
        for phase in ['train', 'val']:
            current_dataset += 1
            if VERBOSE:
                print(f"\n[{current_dataset}/{total_datasets}] Generating data for SNR={snr}, block_length={val_block_length}, gamma={gamma}, phase={phase}...")
            start_time = time.time()
            
            # Calculate transmission_length (same as in trainer.py)
            transmission_length = val_block_length + 8 * hyperparams['n_symbols']

            # Must match the per-phase settings used by Trainer.initialize_channel_data,
            # otherwise the pre-generated cache entries never get reused.
            frames = hyperparams['train_frames'] if phase == 'train' else hyperparams['val_frames']

            # Create temporary dataset to trigger cache generation
            dataset = ChannelModelDataset(
                channel_type=hyperparams['channel_type'],
                block_length=val_block_length,
                transmission_length=transmission_length,
                words=frames * hyperparams['subframes_in_frame'],
                memory_length=hyperparams['memory_length'],
                channel_coefficients=hyperparams['channel_coefficients'],
                random=RandomState(),
                word_rand_gen=RandomState(),
                noisy_est_var=hyperparams['noisy_est_var'],
                fading_taps_type=hyperparams['fading_taps_type'],
                use_ecc=True,
                n_symbols=hyperparams['n_symbols'],
                fading_in_channel=hyperparams['fading_in_channel'],
                fading_in_decoder=hyperparams['fading_in_decoder'],
                phase=phase
            )
            
            # Trigger data generation/loading (this will cache if not exists)
            _ = dataset.__getitem__([snr], gamma)
            
            elapsed = time.time() - start_time
            if VERBOSE:
                print(f"   ✓ Completed in {elapsed:.2f}s")
            
            del dataset
            gc.collect()
    
    # Get cache stats after
    stats_after = cache.get_cache_stats()
    if VERBOSE:
        print(f"\n{'='*70}")
        print(f"✅ PRE-GENERATION COMPLETE!")
        print(f"📦 Cache after: {stats_after['num_files']} files, {stats_after['total_size_mb']:.2f} MB")
        print(f"🆕 Generated {stats_after['num_files'] - stats_before['num_files']} new cache files")
        print(f"{'='*70}\n")
        
        # Show cache efficiency message
        if stats_after['num_files'] > stats_before['num_files']:
            print("💡 Data cached successfully! Subsequent runs will be much faster.\n")
        else:
            print("💡 All data already cached! No generation needed.\n")

def execute_and_plot(model_name, detector_method, self_supervised, all_curves, current_params, run_over, num_rep, perf_tracker):
    method_name = model_name + "_" + detector_method
    
    if VERBOSE:
        print(f"\n{'─'*70}")
        print(f"🚀 Starting model: {model_name}")
        print(f"   Method: {detector_method}, Reps: {num_rep}")
        print(f"{'─'*70}")
    
    # Start timing the model execution
    perf_tracker.start_timing(model_name)
    if 'trainer' in locals() or 'trainer' in globals():
      del trainer
    trainer = Trainer(
                    model_name=model_name,
                    detector_method=detector_method,
                    self_supervised=self_supervised,
                    weights_dir=os.path.join(WEIGHTS_DIR,
                    f'{method_name}_training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1_{HYPERPARAMS_DICT["channel_coefficients"]}'),
                    **HYPERPARAMS_DICT)
    
    # Move model to device and get model size
    model_size = 0
    if hasattr(trainer.detector, "model"):
        try:
            trainer.detector.model.to(device)
            if VERBOSE:
                print(f"✓ Model '{model_name}' loaded on {device}")
        except Exception as e:
            if VERBOSE:
                print(f"✗ Warning: Could not move model to {device}: {e}")
        model_parameters = filter(lambda p: p.requires_grad, trainer.detector.model.parameters())
        model_size = sum([torch.numel(p) for p in model_parameters])
    
    # Profile computational complexity (MACs/bit, latency) for the complexity analysis
    complexity = None
    if hasattr(trainer.detector, "model"):
        try:
            complexity = profile_model(
                trainer.detector.model,
                transmission_length=HYPERPARAMS_DICT['val_block_length'] + 8 * HYPERPARAMS_DICT['n_symbols'],
                memory_length=HYPERPARAMS_DICT['memory_length'],
                device=device)
        except Exception as e:
            if VERBOSE:
                print(f"✗ Warning: complexity profiling failed for {model_name}: {e}")

    # Free GPU memory before heavy operations
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all operations complete before freeing memory
    
    # Call get_ser_data with device and dtype parameters (backward compatible)
    try:
        ser = get_ser_data(trainer, run_over=run_over, num_of_rep=num_rep, 
                          method_name=method_name + '_' + current_params, 
                          device=device, dtype=dtype, use_amp=use_amp, scaler=scaler)
    except TypeError:
        try:
            # Try with device and dtype only
            ser = get_ser_data(trainer, run_over=run_over, num_of_rep=num_rep, 
                              method_name=method_name + '_' + current_params, 
                              device=device, dtype=dtype)
        except TypeError:
            try:
                # Try with just device parameter
                ser = get_ser_data(trainer, run_over=run_over, num_of_rep=num_rep, 
                                  method_name=method_name + '_' + current_params, device=device)
            except TypeError:
                # Fallback to original signature
                ser = get_ser_data(trainer, run_over=run_over, num_of_rep=num_rep, 
                                  method_name=method_name + '_' + current_params)
    
    # End timing
    run_time = perf_tracker.end_timing(model_name)
    
    # Calculate final SER (mean of the SER values)
    final_ser = np.mean(ser)
    
    # Get bit statistics from trainer
    total_bits = getattr(trainer, 'last_eval_total_bits', 0)
    error_bits = getattr(trainer, 'last_eval_error_bits', 0)
    
    if VERBOSE:
        print(f"\n📊 Results for {model_name}:")
        print(f"   Final SER: {final_ser:.6f}")
        print(f"   Total Bits Sent: {total_bits:,}")
        print(f"   Error Bits: {error_bits:,}")
        print(f"   Correct Bits: {total_bits - error_bits:,}")
        print(f"   Model Size: {model_size:,} parameters")
        print(f"   Run Time: {run_time:.2f}s")
    
    # Record metrics. ser is a per-block array, so it also yields a confidence
    # interval - needed to tell whether model gaps are larger than the noise.
    ser_reps = np.asarray(ser).reshape(-1).tolist()
    train_samples = (HYPERPARAMS_DICT['train_frames']
                     * HYPERPARAMS_DICT['subframes_in_frame']
                     * HYPERPARAMS_DICT['train_minibatch_num'])
    perf_tracker.record_metrics(
        model_name=model_name,
        snr=HYPERPARAMS_DICT['curr_SNR'],
        final_ser=final_ser,
        model_size=model_size,
        run_time=run_time,
        config={
            'memory_length': HYPERPARAMS_DICT['memory_length'],
            'noisy_est_var': HYPERPARAMS_DICT['noisy_est_var'],
            'train_samples': train_samples,
            'channel_coefficients': HYPERPARAMS_DICT['channel_coefficients'],
            'n_symbols': HYPERPARAMS_DICT['n_symbols'],
            'val_block_length': HYPERPARAMS_DICT['val_block_length'],
            'pilots_num': HYPERPARAMS_DICT['pilots_num'],
            'detector_method': detector_method,
        },
        complexity=complexity,
        ser_reps=ser_reps
    )
    
    # Print utilization after model completes
    if VERBOSE:
        print(f"\n📊 Utilization after {model_name} completed:")
        print_utilization()
    
    all_curves.append((ser, model_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))
    if VERBOSE:
        print(f"✅ {model_name} completed!\n")


HYPERPARAMS_DICT = {
                    'noisy_est_var': 0,  # >0 injects CSI uncertainty into taps 2..L
                    'memory_length': 4,  # channel memory (number of taps); n_states = 2**memory_length
                    'fading_taps_type': 1,  # 1 / 2  for time decay only
                    'fading_in_channel': True,
                    'fading_in_decoder': True,
                    'gamma': 0.2,
                    'channel_type': 'ISI_AWGN',
                    'train_frames': 3,  # Training batch size = 3*25 = 75
                    'val_frames': 3,  # Validation batch size = 3*25 = 75
                    'subframes_in_frame': 25,  # up to 25 for cost2100
                    'self_supervised_iterations': 200,
                    'ser_thresh': 0.02,  # ser threshold for online training
                    'train_minibatch_num': 200,  # Increased to compensate for smaller batch size
                    }


# ============================================================================
# Experiment sweep configuration
# ============================================================================
# Each sweep produces one of the studies requested in review. A sweep is a list
# of override dicts applied on top of HYPERPARAMS_DICT; every run records its
# own configuration into the results CSV, so rows remain distinguishable.
#
# IMPORTANT: memory_length and noisy_est_var are part of the data-cache key, so
# changing them correctly triggers regeneration rather than silently reusing
# data generated under a different channel.

# Full SNR ladder: (snr, val_block_length, pilot_length).
# Block length is scaled up at high SNR so enough errors are observed to make
# the BER estimate meaningful.
FULL_SNR_PARAMETERS = [(snr, 120, 25) for snr in range(7, 12)] + \
                      [(12, 120 * 5, 25 * 5),
                       (13, 120 * 5, 25 * 5),
                       (14, 120 * 9, 25 * 9),
                       (15, 120 * 9, 25 * 9),
                       (16, 120 * 9, 25 * 9),
                       (17, 120 * 9, 25 * 9)]

# Reduced ladder for quick runs
SHORT_SNR_PARAMETERS = [(15, 120 * 9, 25 * 9),
                        (16, 120 * 9, 25 * 9),
                        (17, 120 * 9, 25 * 9)]


def build_sweep(sweep_name: str):
    """Return a list of (label, overrides) pairs for the requested study."""
    if sweep_name == 'baseline':
        # Single configuration - the standard 4-tap, perfect-CSI setting
        return [('baseline', {})]

    if sweep_name == 'memory':
        # Reviewer question: is the architecture hard-wired to 4-tap memory?
        # Note the trellis grows as 2**memory_length, so large L is expensive.
        return [(f'memory_L{L}', {'memory_length': L}) for L in (2, 3, 4, 5, 6)]

    if sweep_name == 'imperfect_csi':
        # Reviewer question: how do the detectors behave under CSI uncertainty?
        return [(f'csi_var{v}', {'noisy_est_var': v})
                for v in (0.0, 0.01, 0.05, 0.1, 0.5)]

    if sweep_name == 'training_size':
        # Reviewer question: was the ViterbiNet baseline under-trained? Sweep the
        # training volume until BER saturates. Effective number of training words
        # is train_frames * subframes_in_frame * train_minibatch_num.
        return [(f'train_mb{n}', {'train_minibatch_num': n})
                for n in (25, 50, 100, 200, 400, 800)]

    raise ValueError(f"Unknown sweep '{sweep_name}'. Valid: baseline, memory, "
                     f"imperfect_csi, training_size")


if __name__ == '__main__':
  # Initialize the performance tracker
  perf_tracker = ModelPerformanceTracker()
  
  # Print initial system state
#   print_utilization()
  
  # Start background monitoring (every 30 seconds)
#   start_monitoring(interval=300)
  
  # ==========================================================================
  # Experiment selection
  # ==========================================================================
  # SWEEP: 'baseline' | 'memory' | 'imperfect_csi' | 'training_size'
  SWEEP = 'baseline'
  # Use FULL_SNR_PARAMETERS to reproduce the 7-17 dB benchmark ladder, or
  # SHORT_SNR_PARAMETERS for a quick check.
  parameters = FULL_SNR_PARAMETERS
  NUM_OF_REPS = 9          # evaluation repetitions per configuration
  MC_ITERATIONS = 1        # independent Monte-Carlo restarts

  sweep_configs = build_sweep(SWEEP)

  # Baseline settings shared by every sweep entry
  HYPERPARAMS_DICT['n_symbols'] = 2
  HYPERPARAMS_DICT['channel_coefficients'] = 'cost2100'
  # Match the value the main loop derives, so pre-generated entries are actually reused
  HYPERPARAMS_DICT['fading_in_channel'] = True if HYPERPARAMS_DICT['channel_coefficients'] == 'time_decay' else False

  # Preserve the pristine defaults so each sweep entry starts from a clean slate
  BASE_HYPERPARAMS = dict(HYPERPARAMS_DICT)

  print(f"\n{'='*70}")
  print(f"SWEEP: {SWEEP}  ({len(sweep_configs)} configuration(s))")
  print(f"SNR points: {[p[0] for p in parameters]}")
  print(f"{'='*70}\n")

  # Enable CUDA matmul optimizations if using GPU
  if device.type == "cuda":
      torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
      torch.backends.cuda.matmul.allow_fp16_accumulation = True

  for sweep_label, overrides in sweep_configs:
    print(f"\n{'#'*70}")
    print(f"# CONFIGURATION: {sweep_label}  {overrides if overrides else '(defaults)'}")
    print(f"{'#'*70}\n")

    # Reset to defaults, then apply this configuration's overrides
    HYPERPARAMS_DICT.clear()
    HYPERPARAMS_DICT.update(BASE_HYPERPARAMS)
    HYPERPARAMS_DICT.update(overrides)

    # Pre-generate data for this configuration. Because memory_length and
    # noisy_est_var are part of the cache key, each configuration gets its own
    # cache entries instead of silently reusing another configuration's data.
    pre_generate_data(parameters, HYPERPARAMS_DICT['gamma'], HYPERPARAMS_DICT)

    for mc in range(MC_ITERATIONS):
      print(f"\n{'🔄 '*35}")
      print(f"{'='*70}")
      print(f"ITERATION {mc + 1}/{MC_ITERATIONS}  [{sweep_label}]")
      print(f"{'='*70}")

      # Reduce CPU/GPU pressure between top-level iterations
      gc.collect()
      if device.type == "cuda":
          torch.cuda.empty_cache()

      # main flags
      run_over = 2 # 0 - load plots from previous runs if exists / 1 - load trained weights and start online evaluation / 2 - clear all and start training  from scratch
      plot_by_block = False  # False / True either plot by SNR or by block index
      block_length = 120  # determine the transmission length
      channel_coefficients = HYPERPARAMS_DICT['channel_coefficients']  # 'time_decay' / 'cost2100'
      n_symbol = HYPERPARAMS_DICT['n_symbols']
      # deep learning models list 'ADNN', 'Sionna', 'SionnaPlus', 'Transformer', 'LSTM', 'ViterbiNet'
      # 'Sionna' is the unmodified baseline; keep it in the sweep so SionnaPlus can be
      # reported as a gain/loss against the original architecture.
      # 'SionnaAdd' / 'SionnaSkip' are fusion-variant ablations of SionnaPlus.
      models_list = ['Sionna', 'SionnaPlus', 'Transformer', 'ViterbiNet', 'ClassicViterbi']
      # Ablation study (justifies the fusion design):
      # models_list = ['Sionna', 'ViterbiNet', 'SionnaPlus', 'SionnaAdd', 'SionnaSkip']
      detector_method = 'ModelBased'  # ModelBased / EndToEnd / Statistical
      self_supervised = False  # True / False for online evaluation enablement
      all_curves = []
      snr_values = []
      total_params = len(parameters)
      for idx, (snr, val_block_length, pilot_length) in enumerate(parameters, 1):
        print(f'\n{"─"*70}')
        print(f'📍 Parameter Set [{idx}/{total_params}]  [{sweep_label}]:')
        print(f'   SNR={snr}, val_block_length={val_block_length}, pilots_length={pilot_length}, n_symbols={n_symbol}')
        print(f'   memory_length={HYPERPARAMS_DICT["memory_length"]}, noisy_est_var={HYPERPARAMS_DICT["noisy_est_var"]}')
        print(f'   Iteration: {mc}, block-length={block_length}')
        print(f'{"─"*70}')

        HYPERPARAMS_DICT['curr_SNR'] = snr
        HYPERPARAMS_DICT['val_block_length'] = val_block_length
        HYPERPARAMS_DICT['train_block_length'] = val_block_length
        HYPERPARAMS_DICT['pilots_num'] = pilot_length
        current_params = HYPERPARAMS_DICT['channel_coefficients'] + '_' + str(HYPERPARAMS_DICT['curr_SNR']) + '_' + \
                        str(HYPERPARAMS_DICT['val_block_length']) + '_' + str(HYPERPARAMS_DICT['n_symbols'])
        # Keep weights from different sweep configurations separate
        if sweep_label != 'baseline':
          current_params += '_' + sweep_label

        print(f"\n🤖 Running {len(models_list)} models: {', '.join(models_list)}\n")

        for model_idx, model in enumerate(models_list, 1):
            print(f"   [{model_idx}/{len(models_list)}] Processing model: {model}")
            if model == 'ClassicViterbi':
                execute_and_plot(model, 'Statistical', False, all_curves, current_params, run_over, NUM_OF_REPS, perf_tracker)  # Classic Viterbi Alg with Perfect-CSI
            else:
                execute_and_plot(model, detector_method, self_supervised, all_curves, current_params, run_over, NUM_OF_REPS, perf_tracker)

        # Save metrics after each SNR level
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"model_performance_{sweep_label}_snr_{snr}_{timestamp}.csv"
        perf_tracker.save_to_csv(csv_filename)
        print(f"\n💾 Metrics saved to: {csv_filename}\n")

      # if not plot_by_block:
      #     plot_ser_by_snr(all_curves, snr_values)
      # else:
      #     plot_ser_by_block_index(all_curves, block_length, n_symbol, snr)

      # plot_summary_table(all_curves, models_list, snr_values)

      # Save final metrics after all iterations
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      final_csv = f"model_performance_final_{sweep_label}_mc_{mc}_{timestamp}.csv"
      perf_tracker.save_to_csv(final_csv)
      print(f"\n{'='*70}")
      print(f"💾 Final metrics for iteration {mc+1} saved to: {final_csv}")
      print(f"{'='*70}\n")

  # Export the reviewer-facing summary tables across every configuration
  perf_tracker.save_ber_vs_snr_table()
  perf_tracker.save_complexity_table()

  # Stop monitoring and print final stats
  stop_monitoring()
  print("\n" + "🎉"*35)
  print("="*70)
  print("ALL ITERATIONS COMPLETED!")
  print("="*70)
  print("\n📊 FINAL SYSTEM STATE:")
  print_utilization()

