from torch_geometric.nn import TransformerConv
import torch.nn as nn
import torch.nn.functional as F
import src.constants as const

class GAT(nn.Module):
    def __init__(self, in_channels=2, edge_dim=1, hidden_dim=const.HIDDEN_SIZE, heads=const.HEADS):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            TransformerConv(in_channels, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True)
        )
        for _ in range(const.LAYERS - 1):
            self.layers.append(
                TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True)
            )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * heads, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        out = self.classifier(x)
        return out  
