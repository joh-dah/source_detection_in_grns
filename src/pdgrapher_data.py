#%%
# load data from external/PDGrapher/data/processed/torch_data/real_lognorm/data_forward_A549.pt
import torch

file_path = "external/PDGrapher/data/processed/torch_data/real_lognorm/data_forward_A549.pt"

data = torch.load(file_path)
print(f"Data loaded successfully from {file_path}")
data
# %%
