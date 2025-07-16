"""Global constants for the project."""
import yaml

params = yaml.full_load(open("params.yaml", "r"))

# General
MODEL = params["model"]
MODEL_NAME = params["model_name"]
DATA_PATH = "data"
PROCESSED_PATH = f"{DATA_PATH}/processed/{MODEL}"
RAW_PATH = f"{DATA_PATH}/raw"
RAW_EDGE_INDEX_PATH = f"{DATA_PATH}/edge_index.pt"
PROCESSED_EDGE_INDEX_PATH = f"{PROCESSED_PATH}/edge_index.pt"
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
dc = params["data_creation"]
NORMALIZE_DATA = dc["normalize_data"]
N_SAMPLES = dc["n_samples"]
TRAINING_SHARE = dc["training_share"]

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
    "tiny": 4,
    "tp53": 30,
}
NETWORK = params["network"]  # "tiny" or "tp53"

N_NODES = network_dict[NETWORK]
GCNSI_N_FEATURES = 2
