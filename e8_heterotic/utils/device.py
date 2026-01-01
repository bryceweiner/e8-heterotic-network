"""
Device selection utilities for E8×E8 heterotic network computations.

This module provides functions for intelligently selecting the best available
compute device (CUDA, MPS, CPU) and handling device-specific optimizations.
"""

import torch
import warnings
from typing import Optional, Dict, Any

def get_available_devices() -> Dict[str, bool]:
    """
    Check which compute devices are available.

    Returns:
        dict: Dictionary with device availability status
    """
    devices = {
        'cuda': torch.cuda.is_available(),
        'mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'cpu': True  # CPU is always available
    }

    # Additional CUDA info
    if devices['cuda']:
        devices['cuda_count'] = torch.cuda.device_count()
        devices['cuda_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        devices['cuda_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else 0

    return devices

def get_optimal_device(preferred_device: Optional[str] = None,
                      memory_threshold_gb: float = 4.0) -> torch.device:
    """
    Get the optimal compute device based on availability and preferences.

    Priority order (when no preference specified):
    1. CUDA (if sufficient memory)
    2. MPS (Apple Silicon)
    3. CPU

    Args:
        preferred_device: Preferred device ('cuda', 'mps', 'cpu', or None for auto)
        memory_threshold_gb: Minimum GPU memory required for CUDA

    Returns:
        torch.device: Selected device
    """
    available = get_available_devices()

    # If preferred device is specified, try to use it
    if preferred_device:
        preferred_device = preferred_device.lower()

        if preferred_device == 'cuda' and available['cuda']:
            if available.get('cuda_memory', 0) >= memory_threshold_gb:
                return torch.device('cuda:0')
            else:
                warnings.warn(f"CUDA device has insufficient memory ({available.get('cuda_memory', 0):.1f} GB < {memory_threshold_gb} GB). Falling back to CPU.")
                return torch.device('cpu')

        elif preferred_device == 'mps' and available['mps']:
            return torch.device('mps')

        elif preferred_device == 'cpu':
            return torch.device('cpu')

        else:
            warnings.warn(f"Preferred device '{preferred_device}' not available. Falling back to automatic selection.")
            return get_optimal_device(None, memory_threshold_gb)

    # Automatic device selection
    if available['cuda'] and available.get('cuda_memory', 0) >= memory_threshold_gb:
        return torch.device('cuda:0')

    if available['mps']:
        return torch.device('mps')

    return torch.device('cpu')

def get_sparse_safe_device() -> torch.device:
    """
    Get a device that is safe for sparse tensor operations.

    CUDA and CPU support sparse tensors well. MPS has limitations with sparse tensors,
    so we fall back to CPU for MPS systems when sparse operations are needed.

    Returns:
        torch.device: Device safe for sparse operations
    """
    available = get_available_devices()

    if available['cuda']:
        return torch.device('cuda:0')

    # CPU is always safe for sparse operations
    return torch.device('cpu')

def is_sparse_supported(device: torch.device) -> bool:
    """
    Check if sparse tensor operations are well supported on the given device.

    Args:
        device: PyTorch device to check

    Returns:
        bool: True if sparse operations are well supported
    """
    # CUDA and CPU have good sparse support
    if device.type in ['cuda', 'cpu']:
        return True

    # MPS has limited sparse support
    if device.type == 'mps':
        return False

    # Unknown device - assume not supported
    return False

def get_device_info(device: torch.device) -> Dict[str, Any]:
    """
    Get detailed information about a device.

    Args:
        device: PyTorch device

    Returns:
        dict: Device information
    """
    info = {
        'type': device.type,
        'index': device.index if device.index is not None else None,
    }

    if device.type == 'cuda':
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            info.update({
                'name': props.name,
                'memory_total_gb': props.total_memory / (1024**3),
                'memory_free_gb': torch.cuda.mem_get_info(device)[0] / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count,
                'sparse_supported': True
            })
        else:
            info['error'] = 'CUDA not available'

    elif device.type == 'mps':
        info.update({
            'sparse_supported': False,
            'note': 'MPS backend has limited sparse tensor support'
        })

    elif device.type == 'cpu':
        import psutil
        try:
            memory = psutil.virtual_memory()
            info.update({
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'cpu_count': psutil.cpu_count(),
                'sparse_supported': True
            })
        except ImportError:
            info['note'] = 'psutil not available for CPU info'

    return info

def move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Safely move a tensor to a device, handling potential issues.

    Args:
        tensor: Input tensor
        device: Target device

    Returns:
        torch.Tensor: Tensor on target device
    """
    try:
        return tensor.to(device)
    except Exception as e:
        warnings.warn(f"Failed to move tensor to {device}: {e}. Keeping on current device.")
        return tensor

def get_memory_efficient_device(batch_size: int,
                               feature_dim: int,
                               model_params: int,
                               memory_threshold_gb: float = 2.0) -> torch.device:
    """
    Select device based on estimated memory requirements.

    Args:
        batch_size: Batch size for computations
        feature_dim: Feature dimension
        model_params: Number of model parameters
        memory_threshold_gb: Minimum free memory required

    Returns:
        torch.device: Selected device
    """
    # Rough memory estimate (in GB)
    # This is a very rough estimate - actual memory usage depends on many factors
    estimated_memory = (
        batch_size * feature_dim * 4 +  # activations (float32)
        model_params * 4 * 2            # parameters + gradients
    ) / (1024**3)

    available = get_available_devices()

    # Check CUDA memory
    if available['cuda']:
        cuda_memory_free = available.get('cuda_memory', 0)
        if cuda_memory_free >= estimated_memory + memory_threshold_gb:
            return torch.device('cuda:0')

    # Check system memory for CPU/MPS
    try:
        import psutil
        system_memory_free = psutil.virtual_memory().available / (1024**3)

        if system_memory_free >= estimated_memory + memory_threshold_gb:
            if available['mps']:
                return torch.device('mps')
            else:
                return torch.device('cpu')
    except ImportError:
        pass

    # Fallback to safest option
    return torch.device('cpu')

def print_device_info(device: Optional[torch.device] = None):
    """
    Print comprehensive device information.

    Args:
        device: Specific device to print info for (default: current optimal)
    """
    if device is None:
        device = get_optimal_device()

    print(f"\nDevice Information for {device}:")
    print("=" * 50)

    info = get_device_info(device)
    for key, value in info.items():
        if key == 'memory_free_gb' and isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f} GB")
        elif key == 'memory_total_gb' and isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f} GB")
        else:
            print(f"  {key}: {value}")

    # Print all available devices
    print(f"\nAvailable Devices:")
    available = get_available_devices()
    for dev_type, is_available in available.items():
        if dev_type in ['cuda', 'mps', 'cpu']:
            status = "✓ Available" if is_available else "✗ Not available"
            print(f"  {dev_type.upper()}: {status}")

            if dev_type == 'cuda' and is_available:
                cuda_info = available.get('cuda_count', 0)
                print(f"    Devices: {cuda_info}")

    print()

# =============================================================================
# E8×E8 SPECIFIC DEVICE OPTIMIZATIONS
# =============================================================================

def get_e8_computation_device(matrix_size: int = 496) -> torch.device:
    """
    Get optimal device for E8×E8 computations based on matrix size.

    For the 496×496 E8×E8 adjacency matrix, larger GPUs are preferred.

    Args:
        matrix_size: Size of matrices to be computed (default: 496 for E8×E8)

    Returns:
        torch.device: Optimal device for E8×E8 computations
    """
    # Estimate memory requirements for E8×E8 computations
    # Matrix size is 496×496, stored as float32
    matrix_memory_gb = (matrix_size ** 2 * 4) / (1024**3)  # ~1GB for 496×496 float32

    # Add overhead for computations
    total_memory_needed = matrix_memory_gb * 3  # matrices + workspace

    return get_optimal_device(memory_threshold_gb=total_memory_needed)

def setup_e8_computation_environment():
    """
    Setup optimal environment for E8×E8 computations.

    This function configures PyTorch settings for optimal E8×E8 performance.

    Returns:
        dict: Environment configuration
    """
    config = {
        'device': get_e8_computation_device(),
        'use_sparse': False,
        'precision': torch.float32,
        'warnings': []
    }

    # Determine if sparse operations should be used
    config['use_sparse'] = is_sparse_supported(config['device'])

    if not config['use_sparse']:
        config['warnings'].append(
            f"Sparse operations not well supported on {config['device'].type}. "
            "Using dense matrices instead."
        )

    # Set precision based on device capabilities
    if config['device'].type == 'cuda':
        # CUDA can handle float64 if needed
        config['precision'] = torch.float32  # Use float32 for speed
    else:
        config['precision'] = torch.float32

    # Print configuration
    print("E8×E8 Computation Environment Setup:")
    print(f"  Device: {config['device']}")
    print(f"  Sparse operations: {'Enabled' if config['use_sparse'] else 'Disabled'}")
    print(f"  Precision: {config['precision']}")

    if config['warnings']:
        for warning in config['warnings']:
            print(f"  Warning: {warning}")

    return config

if __name__ == "__main__":
    # Print comprehensive device information
    print_device_info()

    # Setup E8×E8 environment
    print("\nSetting up E8×E8 computation environment:")
    config = setup_e8_computation_environment()