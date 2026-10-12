"""
Universal Genome Transformation System for Evolutionary Neural Networks

- Supports arbitrary fully connected architectures
- Handles width/depth changes, neuron expansion, and layer addition/removal
- Tracks copied vs. new weights for progressive unfreezing
- Pure numpy, project-agnostic
"""
import numpy as np
from typing import List, Tuple, Dict, Any


def decode_genome(genome: np.ndarray, arch: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Decode a flat genome into weight matrices and bias vectors for a given architecture.
    arch: List of layer sizes, e.g. [in, h1, h2, ..., out]
    Returns: (weights, biases) where each is a list of np.ndarray
    """
    weights = []
    biases = []
    idx = 0
    for i in range(len(arch) - 1):
        w_shape = (arch[i], arch[i+1])
        b_shape = (arch[i+1],)
        w_size = np.prod(w_shape)
        b_size = arch[i+1]
        w = genome[idx:idx+w_size].reshape(w_shape)
        idx += w_size
        b = genome[idx:idx+b_size]
        idx += b_size
        weights.append(w)
        biases.append(b)
    return weights, biases


def encode_genome(weights: List[np.ndarray], biases: List[np.ndarray]) -> np.ndarray:
    """
    Encode weights and biases into a flat genome.
    """
    flat = []
    for w, b in zip(weights, biases):
        flat.append(w.flatten())
        flat.append(b.flatten())
    return np.concatenate(flat)


def transform_genome(
    source_genome: np.ndarray,
    source_arch: List[int],
    target_arch: List[int],
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Transform a genome from source_arch to target_arch.
    Returns: (new_genome, info_dict)
    info_dict contains mask arrays and stats for progressive unfreezing.
    """
    rng = np.random.default_rng(seed)
    src_w, src_b = decode_genome(source_genome, source_arch)
    tgt_w = []
    tgt_b = []
    copy_mask_w = []
    copy_mask_b = []
    total_params = 0
    copied_params = 0
    for i in range(len(target_arch) - 1):
        tgt_shape = (target_arch[i], target_arch[i+1])
        tgt_b_shape = (target_arch[i+1],)
        # If this layer exists in source
        if i < len(source_arch) - 1:
            src_shape = (source_arch[i], source_arch[i+1])
            src_b_shape = (source_arch[i+1],)
            # Copy overlapping part
            w = np.zeros(tgt_shape, dtype=np.float32)
            b = np.zeros(tgt_b_shape, dtype=np.float32)
            mask_w = np.zeros(tgt_shape, dtype=bool)
            mask_b = np.zeros(tgt_b_shape, dtype=bool)
            min_rows = min(src_shape[0], tgt_shape[0])
            min_cols = min(src_shape[1], tgt_shape[1])
            w[:min_rows, :min_cols] = src_w[i][:min_rows, :min_cols]
            b[:min_cols] = src_b[i][:min_cols]
            mask_w[:min_rows, :min_cols] = True
            mask_b[:min_cols] = True
            # Initialize new neurons
            if tgt_shape != src_shape:
                w[min_rows:, :] = rng.normal(0, 0.01, size=(tgt_shape[0]-min_rows, tgt_shape[1]))
                w[:, min_cols:] = rng.normal(0, 0.01, size=(tgt_shape[0], tgt_shape[1]-min_cols))
                b[min_cols:] = rng.normal(0, 0.01, size=(tgt_b_shape[0]-min_cols,))
            tgt_w.append(w)
            tgt_b.append(b)
            copy_mask_w.append(mask_w)
            copy_mask_b.append(mask_b)
            total_params += np.prod(tgt_shape) + tgt_b_shape[0]
            copied_params += np.sum(mask_w) + np.sum(mask_b)
        else:
            # New layer
            w = rng.normal(0, 0.01, size=tgt_shape).astype(np.float32)
            b = rng.normal(0, 0.01, size=tgt_b_shape).astype(np.float32)
            mask_w = np.zeros(tgt_shape, dtype=bool)
            mask_b = np.zeros(tgt_b_shape, dtype=bool)
            tgt_w.append(w)
            tgt_b.append(b)
            copy_mask_w.append(mask_w)
            copy_mask_b.append(mask_b)
            total_params += np.prod(tgt_shape) + tgt_b_shape[0]
    info = {
        'copy_mask_w': copy_mask_w,
        'copy_mask_b': copy_mask_b,
        'percent_copied': 100.0 * copied_params / total_params if total_params > 0 else 0.0,
        'total_params': total_params,
        'copied_params': copied_params
    }
    return encode_genome(tgt_w, tgt_b), info


def analyze_transfer(source_arch: List[int], target_arch: List[int]) -> Dict[str, Any]:
    """
    Report which layers will be copied, expanded, truncated, or newly initialized.
    """
    report = []
    n_src = len(source_arch) - 1
    n_tgt = len(target_arch) - 1
    for i in range(max(n_src, n_tgt)):
        if i < n_src and i < n_tgt:
            src_shape = (source_arch[i], source_arch[i+1])
            tgt_shape = (target_arch[i], target_arch[i+1])
            if src_shape == tgt_shape:
                status = 'copied'
            else:
                status = 'expanded' if (tgt_shape[0] > src_shape[0] or tgt_shape[1] > src_shape[1]) else 'truncated'
            report.append({'layer': i, 'source': src_shape, 'target': tgt_shape, 'status': status})
        elif i < n_src:
            report.append({'layer': i, 'source': (source_arch[i], source_arch[i+1]), 'target': None, 'status': 'removed'})
        else:
            report.append({'layer': i, 'source': None, 'target': (target_arch[i], target_arch[i+1]), 'status': 'new'})
    return {'layers': report}


def validate_transfer(source_network, target_network, test_inputs: np.ndarray, atol=1e-5) -> Dict[str, Any]:
    """
    Compare outputs of two networks on test_inputs.
    Returns dict with similarity stats.
    Assumes both networks have a .forward(x) method or are callable.
    """
    src_out = source_network(test_inputs)
    tgt_out = target_network(test_inputs)
    diff = np.abs(src_out - tgt_out)
    return {
        'mean_abs_diff': float(np.mean(diff)),
        'max_abs_diff': float(np.max(diff)),
        'similar': bool(np.allclose(src_out, tgt_out, atol=atol))
    }
