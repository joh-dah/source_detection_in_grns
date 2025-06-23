from torch_geometric.nn import global_mean_pool, TransformerConv
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, in_channels=2, edge_dim=1, hidden_dim=64, num_classes=30, num_layers=2, heads=2):
        super().__init__()

        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(
            TransformerConv(in_channels, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True)
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True)
            )

        self.pool = global_mean_pool

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * heads, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # print shape of x, edge_index, edge_attr, batch
        print(f"x shape: {x.shape}, edge_index shape: {edge_index.shape}, edge_attr shape: {edge_attr.shape}, batch shape: {batch.shape}")

        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = self.pool(x, batch)
        out = self.classifier(x)
        return out
