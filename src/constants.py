"""Global constants for the project."""
import yaml
import argparse
import os
from pathlib import Path
from typing import Dict, Any
from src.data_utils import get_shared_data_path, get_experiment_data_path

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
dc = params["data_creation"]
NETWORK = dc["network"]
SEED = params["seed"]
if MODEL not in ["gat", "pdgrapher", "gcnsi", "pdgraphernognn"]:
    raise ValueError(f"Invalid model type: {MODEL}. Expected one of ['gat', 'pdgrapher', 'gcnsi', 'pdgraphernognn'].")
MODEL_NAME = f"{params["model_name"]}_{EXPERIMENT}"

# Path structure: shared data for same data_creation params, experiment-specific for perturbations
SHARED_DATA_PATH = get_shared_data_path(params["data_creation"], NETWORK, SEED)
EXPERIMENT_DATA_PATH = get_experiment_data_path(
    params["data_creation"], 
    params.get("graph_perturbation", {}), 
    NETWORK, 
    SEED, 
    EXPERIMENT
)

# Shared paths (raw data and splits only)
RAW_PATH = f"{SHARED_DATA_PATH}/raw"
SPLITS_PATH = f"{SHARED_DATA_PATH}/splits/splits.pt"

# Experiment-specific paths (after graph perturbation)
EXPERIMENT_RAW_PATH = f"{EXPERIMENT_DATA_PATH}/raw"
EXPERIMENT_PROCESSED_PATH = f"{EXPERIMENT_DATA_PATH}/processed/{MODEL}"
EXPERIMENT_EDGE_INDEX_PATH = f"{EXPERIMENT_PROCESSED_PATH}/edge_index.pt"

# Data processing is experiment-specific (uses perturbed graph)
PROCESSED_PATH = EXPERIMENT_PROCESSED_PATH
PROCESSED_EDGE_INDEX_PATH = f"{PROCESSED_PATH}/edge_index.pt"

# Backwards compatibility: DATA_PATH points to experiment-specific path for graph perturbation
DATA_PATH = EXPERIMENT_DATA_PATH

# Other paths
TOPO_PATH = "topos"
MODEL_PATH = "models"
FIGURES_PATH = "figures"
REPORT_PATH = "reports"
ON_CLUSTER = params["on_cluster"]

if ON_CLUSTER:
    N_CORES = 64
else:
    N_CORES = 2

# Data Creation
NORMALIZE_DATA = dc["normalize_data"]
N_SAMPLES = dc["n_samples"]
TIME_STEPS = dc.get("time_steps", None)
TRAINING_SHARE = dc["training_share"]
REMOVE_NEAR_DUPLICATES = dc["remove_near_duplicates"]


# Graph Perturbation (moved from data_creation)
gp = params.get("graph_perturbation", {})
GRAPH_NOISE = gp.get("noise", {"missing_edges": 0, "wrong_edges": 0})
RANDOM_GRAPH = gp.get("random_graph", False)

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
    "dorothea_290": 290,
    "dorothea_500_550": 500,
    "dorothea_500_600": 500,
    "dorothea_500_700": 500,
    "dorothea_500_800": 500,
    "dorothea_500_900": 500,
    "dorothea_500_1000": 500,
    "dorothea_500_1250": 500,
    "dorothea_500_1500": 500,
    "dorothea_500_2000": 500,
    "dorothea_500_2500": 500,
    "dorothea_500_3000": 500,
    "dorothea_500_4000": 500,
    "dorothea_727_sparse": 727,
    "dorothea_882_sparse": 882,
    "dorothea_1000": 1000,
    "dorothea_1000_sparse": 1000,
    "dorothea_2000_sparse": 2000,
    "pdgrapher_grn": 10716,
    "random_10716_300000": 10716,
}


N_NODES = network_dict[NETWORK]
GCNSI_N_FEATURES = 2

# Gene-specific metrics
gene_metrics = params.get("gene_specific_metrics", {})
GENE_METRICS_ENABLED = gene_metrics.get("enabled", False)
GENES_OF_INTEREST = gene_metrics.get("genes_of_interest", [])