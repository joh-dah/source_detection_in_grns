"""Global constants for the project."""
import yaml

params = yaml.full_load(open("params.yaml", "r"))

# General
MODEL = params["model"]  # "GCNR" or "GCNSI"
GCNR_LAYER_TYPE = params["GCNR_layer_type"]  # "GCN" or "GAT"
MODEL_NAME = params["model_name"]  # defins
DATA_PATH = "data"
TOPO_PATH = "topos"
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
MODEL_PATH = "models"
FIGURES_PATH = "figures"
ROC_PATH = "roc"
REPORT_PATH = "reports"
ON_CLUSTER = params["on_cluster"]
N_CORES = 2

# Data Creation
dc = params["data_creation"]
TRAINING_SIZE = dc["training_size"]
VALIDATION_SIZE = dc["validation_size"]
SMALL_INPUT = dc["small_input"]  # "true" or "false"
N_SOURCES = dc["n_sources"]
NORMALIZE_DATA = dc["normalize_data"]

# Training
training = params["training"]
EPOCHS = training["epochs"]
LEARNING_RATE = training["learning_rate"]
DROPOUT = training["dropout"]
HIDDEN_SIZE = training["hidden_size"]
LAYERS = training["layers"]
ALPHA = training["alpha"]
WEIGHT_DECAY = training["weight_decay"]
BATCH_SIZE = training["batch_size"]
USE_LOG_LOSS = training["useLogLoss"]
SUBSAMPLE = training["subsample"]
CLASS_WEIGHTING = training["class_weighting"]
GRAPH_WEIGHTING = training["graph_weighting"]

if SMALL_INPUT:
    GCNSI_N_FEATURES = 2
    GCNR_N_FEATURES = 2
else:
    GCNSI_N_FEATURES = 4
    GCNR_N_FEATURES = 4

# Visualization
SEED = params["visualization"]["seed"]
