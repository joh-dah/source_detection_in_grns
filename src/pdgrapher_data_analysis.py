# load data from external/PDGrapher/data/processed/torch_data/real_lognorm/data_forward_A549.pt
import torch

file_path = "data/processed/splits/splits.pt"
splits = torch.load(file_path, weights_only=False)
print(f"Splits loaded successfully from {file_path}")

file_path = "external/PDGrapher/data/processed/torch_data/real_lognorm/data_forward_A549.pt"
data_forward = torch.load(file_path, weights_only=False)
print(f"Data loaded successfully from {file_path}")
# print length of data_forward
print(f"Length of data_forward: {len(data_forward)}")

file_path = "external/PDGrapher/data/processed/torch_data/real_lognorm/data_backward_A549.pt"
data_backward = torch.load(file_path, weights_only=False)
print(f"Data loaded successfully from {file_path}")
# print length of data_backward
print(f"Length of data_backward: {len(data_backward)}")
data_backward

file_path = "external/PDGrapher/data/splits/genetic/A549/random/1fold/splits.pt"
splits = torch.load(file_path, weights_only=False)
print(f"Splits loaded successfully from {file_path}")

file_path = "external/PDGrapher/data/processed/torch_data/real_lognorm/edge_index_A549.pt"
edge_index = torch.load(file_path, weights_only=False)
print(f"Edge index loaded successfully from {file_path}")