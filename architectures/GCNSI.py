import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import LayerNorm, Dropout
import src.constants as const


class GCNSI(torch.nn.Module):
    def __init__(self):
        super(GCNSI, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs.append(GCNConv(const.GCNSI_N_FEATURES, const.HIDDEN_SIZE))
        self.norms.append(LayerNorm(const.HIDDEN_SIZE))

        for _ in range(1, const.LAYERS):
            self.convs.append(GCNConv(const.HIDDEN_SIZE, const.HIDDEN_SIZE))
            self.norms.append(LayerNorm(const.HIDDEN_SIZE))

        self.dropout = Dropout(p=const.DROPOUT)
        self.classifier = torch.nn.Linear(const.HIDDEN_SIZE, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        return self.classifier(x)
