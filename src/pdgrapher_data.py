# load data from external/PDGrapher/data/processed/torch_data/real_lognorm/data_forward_A549.pt
import torch

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