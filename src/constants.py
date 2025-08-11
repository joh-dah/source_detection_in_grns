"""Global constants for the project."""
import yaml
import argparse
import os
from pathlib import Path
from typing import Dict, Any

def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(model_type: str = None, experiment: str = None, config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from multiple YAML files.
    
    Args:
        model_type: Model type (gat, pdgrapher, gcnsi)
        experiment: Specific experiment config name
        config_path: Direct path to config file (overrides other options)
    
    Returns:
        Merged configuration dictionary
    """
    if config_path:
        # Load directly from specified file
        return yaml.safe_load(open(config_path))
    
    config_dir = Path("configs")
    
    # Load shared config
    shared_config_path = config_dir / "shared.yaml"
    if not shared_config_path.exists():
        raise FileNotFoundError(f"Shared config not found: {shared_config_path}")
    
    config = yaml.safe_load(open(shared_config_path))
    
    # Load model-specific config if provided
    if model_type:
        model_config_path = config_dir / "models" / f"{model_type.lower()}.yaml"
        if model_config_path.exists():
            model_config = yaml.safe_load(open(model_config_path))
            config = deep_merge(config, model_config)
        else:
            print(f"Warning: Model config not found: {model_config_path}")
    
    # Load experiment config if provided
    if experiment:
        exp_config_path = config_dir / "experiments" / f"{experiment}.yaml"
        if exp_config_path.exists():
            exp_config = yaml.safe_load(open(exp_config_path))
            # add experiment name to config
            exp_config["experiment"] = experiment
            config = deep_merge(config, exp_config)
        else:
            print(f"Warning: Experiment config not found: {exp_config_path}")
    
    return config

def get_config_from_args():
    """Get configuration based on command line arguments or environment variables."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)  # Don't interfere with other parsers
    parser.add_argument("--model", type=str, help="Model type (gat, pdgrapher, gcnsi)")
    parser.add_argument("--experiment", type=str, help="Experiment configuration name")
    parser.add_argument("--config", type=str, help="Direct path to config file")
    
    args, unknown = parser.parse_known_args()
    
    # Get from environment variables if not provided via args
    model_type = args.model or os.environ.get('MODEL_TYPE')
    experiment = args.experiment or os.environ.get('EXPERIMENT_NAME')
    config_path = args.config or os.environ.get('CONFIG_PATH')
    
    # Fallback to legacy params.yaml if no model specified
    if not model_type and not config_path:
        legacy_params_path = Path("params.yaml")
        if legacy_params_path.exists():
            print("Warning: Using legacy params.yaml. Consider migrating to new config structure.")
            return yaml.safe_load(open(legacy_params_path))
        else:
            raise ValueError("No model type specified and no legacy params.yaml found")
    
    return load_config(model_type, experiment, config_path)

# Load configuration
params = get_config_from_args()

# General
MODEL = params["model"].lower()
EXPERIMENT = params.get("experiment", "unknown")
if MODEL not in ["gat", "pdgrapher", "gcnsi"]:
    raise ValueError(f"Invalid model type: {MODEL}. Expected one of ['gat', 'pdgrapher', 'gcnsi'].")
MODEL_NAME = f"{params["model_name"]}_{EXPERIMENT}"
DATA_PATH = f"data/{EXPERIMENT}"
PROCESSED_PATH = f"{DATA_PATH}/processed/{MODEL}"
SPLITS_PATH = f"{DATA_PATH}/splits/splits.pt"
RAW_PATH = f"{DATA_PATH}/raw"
RAW_EDGE_INDEX_PATH = f"{DATA_PATH}/edge_index.pt"
PROCESSED_EDGE_INDEX_PATH = f"{PROCESSED_PATH}/edge_index.pt"
TOPO_PATH = "topos"
MODEL_PATH = "models"
FIGURES_PATH = "figures"
REPORT_PATH = "reports"
ON_CLUSTER = params["on_cluster"]
SEED = params["seed"]

if ON_CLUSTER:
    N_CORES = 32
else:
    N_CORES = 2

# Data Creation
dc = params["data_creation"]
NORMALIZE_DATA = dc["normalize_data"]
N_SAMPLES = dc["n_samples"]
TRAINING_SHARE = dc["training_share"]
GRAPH_NOISE = dc["graph_noise"]

# Training
training = params["training"]
EPOCHS = training["epochs"]
LEARNING_RATE = training["learning_rate"]
DROPOUT = training["dropout"]
HIDDEN_SIZE = training["hidden_size"]
HEADS = training["heads"]
LAYERS = training["layers"]
ALPHA = training["alpha"]
WEIGHT_DECAY = training["weight_decay"]
BATCH_SIZE = training["batch_size"]
SUBSAMPLE = training["subsample"]
CLASS_WEIGHTING = training["class_weighting"]
GRAPH_WEIGHTING = training["graph_weighting"]

network_dict = {
    "custom_4": 4,
    "custom_10": 10,
    "tp53_30": 30,
    "dorothea_39": 39,
    "dorothea_60": 60,
    "dorothea_99": 99,
    "dorothea_150": 150,
}
NETWORK = params["network"]

N_NODES = network_dict[NETWORK]
GCNSI_N_FEATURES = 2

# Gene-specific metrics
gene_metrics = params.get("gene_specific_metrics", {})
GENE_METRICS_ENABLED = gene_metrics.get("enabled", False)
GENES_OF_INTEREST = gene_metrics.get("genes_of_interest", [])