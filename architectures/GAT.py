from torch_geometric.nn import global_mean_pool, TransformerConv
import torch.nn as nn
import torch.nn.functional as F
import src.constants as const

class GAT(nn.Module):
    def __init__(self, in_channels=2, edge_dim=1, hidden_dim=const.HIDDEN_SIZE, num_layers=const.LAYERS, heads=4):
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
            nn.Linear(128, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        out = self.classifier(x) 
        return out  
