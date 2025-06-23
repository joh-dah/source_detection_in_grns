from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=64, num_classes=30, num_layers=2, heads=2):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(SignedGATv2Layer(in_channels, hidden_dim, heads=heads))
        for _ in range(num_layers - 1):
            self.layers.append(SignedGATv2Layer(hidden_dim * heads, hidden_dim, heads=heads))

        self.pool = global_mean_pool  # you can swap with attention pooling if needed

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * heads, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = self.pool(x, batch)  # [batch_size, hidden_dim * heads]
        out = self.classifier(x)
        return out  # logits over classes
