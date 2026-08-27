"""
Computational complexity measurement for the detector models.

Three of the four reviewers asked for a complexity analysis. This module provides
the measurements needed to build a BER-gain-vs-complexity comparison:

  * parameter count
  * multiply-accumulate operations (MACs) per detected bit, measured with
    PyTorch module hooks (no third-party dependency)
  * measured inference latency per bit
  * the analytic Viterbi trellis cost, reported separately from the neural cost

Separating the trellis term from the neural term matters because the trellis
cost grows as 2**memory_length and is identical for every model-based detector,
whereas the neural term is what actually differs between architectures.
"""
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def _conv1d_macs(module: nn.Conv1d, output: torch.Tensor) -> int:
    # each output element costs (in_channels/groups * kernel) MACs
    kernel_ops = module.kernel_size[0] * (module.in_channels // module.groups)
    return int(output.numel() * kernel_ops)


def _convtranspose1d_macs(module: nn.ConvTranspose1d, inputs: torch.Tensor) -> int:
    # transposed conv scatters each input element across the kernel
    kernel_ops = module.kernel_size[0] * (module.out_channels // module.groups)
    return int(inputs.numel() * kernel_ops)


def _linear_macs(module: nn.Linear, output: torch.Tensor) -> int:
    return int(output.numel() * module.in_features)


def count_macs(model: nn.Module, sample_input: torch.Tensor) -> int:
    """Count MACs for a single forward pass over ``sample_input``.

    Uses forward hooks on the layer types that dominate cost in these models
    (Linear, Conv1d, ConvTranspose1d). Element-wise layers (ReLU, LayerNorm,
    Sigmoid) are negligible by comparison and are not counted.
    """
    totals = {'macs': 0}
    handles = []

    def make_hook(fn, use_input):
        def hook(module, inputs, output):
            try:
                totals['macs'] += fn(module, inputs[0] if use_input else output)
            except Exception:
                pass
        return hook

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(_linear_macs, False)))
        elif isinstance(module, nn.Conv1d):
            handles.append(module.register_forward_hook(make_hook(_conv1d_macs, False)))
        elif isinstance(module, nn.ConvTranspose1d):
            handles.append(module.register_forward_hook(make_hook(_convtranspose1d_macs, True)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(sample_input)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return totals['macs']


def measure_latency(model: nn.Module, sample_input: torch.Tensor,
                    repeats: int = 20, warmup: int = 5) -> float:
    """Return median wall-clock seconds for one forward pass.

    Median rather than mean, because scheduler noise produces occasional large
    outliers that would otherwise dominate. Includes CUDA synchronisation so the
    timing reflects completed work rather than queued work.
    """
    was_training = model.training
    model.eval()
    is_cuda = sample_input.is_cuda
    timings = []
    try:
        with torch.no_grad():
            for _ in range(warmup):
                model(sample_input)
            if is_cuda:
                torch.cuda.synchronize()
            for _ in range(repeats):
                start = time.perf_counter()
                model(sample_input)
                if is_cuda:
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
    finally:
        model.train(was_training)

    return float(np.median(timings)) if timings else 0.0


def viterbi_trellis_macs_per_bit(memory_length: int) -> int:
    """Analytic add-compare-select cost per detected bit for a binary trellis.

    The trellis has 2**memory_length states and 2 outgoing transitions per state,
    so each decoded bit costs 2 * 2**memory_length add-compare-select operations.
    This term is shared by every ModelBased detector and by ClassicViterbi.
    """
    return int(2 * (2 ** memory_length))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def profile_model(model: nn.Module, transmission_length: int, memory_length: int,
                  device: torch.device = None,
                  measure_time: bool = True) -> Dict[str, float]:
    """Profile one detector and return complexity metrics normalised per bit.

    The input is shaped exactly as ``Detector.forward`` shapes it before calling
    the model - ``(batch, transmission_length, input_size)`` - so the measured
    cost reflects the real inference path rather than a synthetic one.

    ``ClassicViterbi`` and other parameter-free models are handled gracefully:
    they report zero neural cost and only the analytic trellis term.
    """
    device = device or torch.device('cpu')
    metrics = {
        'params': 0,
        'neural_macs_per_bit': 0.0,
        'trellis_macs_per_bit': float(viterbi_trellis_macs_per_bit(memory_length)),
        'total_macs_per_bit': 0.0,
        'latency_ms_per_bit': 0.0,
    }

    metrics['params'] = count_parameters(model)

    input_size = int(getattr(model, 'input_size', 1) or 1)
    sample = torch.zeros(1, transmission_length, input_size, device=device)

    try:
        total_macs = count_macs(model, sample)
        metrics['neural_macs_per_bit'] = float(total_macs) / max(transmission_length, 1)
    except Exception as e:
        print(f"[Complexity] MAC counting unavailable for {type(model).__name__}: {e}")

    if measure_time:
        try:
            latency = measure_latency(model, sample)
            metrics['latency_ms_per_bit'] = (latency * 1e3) / max(transmission_length, 1)
        except Exception as e:
            print(f"[Complexity] Latency measurement unavailable for {type(model).__name__}: {e}")

    metrics['total_macs_per_bit'] = (metrics['neural_macs_per_bit']
                                     + metrics['trellis_macs_per_bit'])
    return metrics
