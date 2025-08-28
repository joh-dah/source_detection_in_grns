"""
Data utilities for shared data architecture.
Implements content-addressable storage for sharing data between experiments with identical data_creation parameters.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, Any


def get_data_fingerprint(data_creation_params: Dict[Any, Any], network: str, seed: int) -> str:
    """
    Create a unique fingerprint for data_creation configuration.
    Experiments with identical fingerprints can share raw data and splits.
    
    Args:
        data_creation_params: The data_creation section from config
        network: Network name (e.g., "dorothea_290")
        seed: Random seed
        
    Returns:
        SHA256 hash string (first 12 characters for readability)
    """
    # Create canonical representation for hashing
    fingerprint_data = {
        "data_creation": data_creation_params,
        "network": network,
        "seed": seed
    }
    
    # Convert to canonical JSON string (sorted keys, no whitespace)
    canonical_json = json.dumps(fingerprint_data, sort_keys=True, separators=(',', ':'))
    
    # Create hash
    hash_object = hashlib.sha256(canonical_json.encode('utf-8'))
    return hash_object.hexdigest()[:12]  # First 12 chars for readability


def get_graph_perturbation_fingerprint(graph_perturbation_params: Dict[Any, Any]) -> str:
    """
    Create a unique fingerprint for graph_perturbation configuration.
    
    Args:
        graph_perturbation_params: The graph_perturbation section from config
        
    Returns:
        SHA256 hash string (first 12 characters for readability)
    """
    # Create canonical representation for hashing
    canonical_json = json.dumps(graph_perturbation_params, sort_keys=True, separators=(',', ':'))
    
    # Create hash
    hash_object = hashlib.sha256(canonical_json.encode('utf-8'))
    return hash_object.hexdigest()[:12]  # First 12 chars for readability


def get_shared_data_path(data_creation_params: Dict[Any, Any], network: str, seed: int) -> str:
    """
    Get the path for shared data based on data_creation configuration.
    
    Args:
        data_creation_params: The data_creation section from config
        network: Network name  
        seed: Random seed
        
    Returns:
        Path string for shared data directory
    """
    fingerprint = get_data_fingerprint(data_creation_params, network, seed)
    return f"data/shared/{fingerprint}"


def get_experiment_data_path(data_creation_params: Dict[Any, Any], graph_perturbation_params: Dict[Any, Any], 
                           network: str, seed: int, experiment: str) -> str:
    """
    Get the path for experiment-specific data.
    
    Args:
        data_creation_params: The data_creation section from config
        graph_perturbation_params: The graph_perturbation section from config
        network: Network name
        seed: Random seed
        experiment: Experiment name
        
    Returns:
        Path string for experiment-specific data directory
    """
    shared_fingerprint = get_data_fingerprint(data_creation_params, network, seed)
    graph_fingerprint = get_graph_perturbation_fingerprint(graph_perturbation_params)
    return f"data/experiments/{shared_fingerprint}/{graph_fingerprint}/{experiment}"


def data_exists(data_path: str) -> bool:
    """
    Check if data already exists at the given path.
    
    Args:
        data_path: Path to check for existing data
        
    Returns:
        True if data exists and contains raw data files
    """
    raw_path = Path(data_path) / "raw"
    if not raw_path.exists():
        return False
    
    # Check if raw directory contains .pt files
    raw_files = list(raw_path.glob("*.pt"))
    return len(raw_files) > 0


def splits_exist(data_path: str) -> bool:
    """
    Check if splits already exist at the given path.
    
    Args:
        data_path: Path to check for existing splits
        
    Returns:
        True if splits file exists
    """
    splits_path = Path(data_path) / "splits" / "splits.pt"
    return splits_path.exists()
