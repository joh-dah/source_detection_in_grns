"""
PDGrapher-inspired architecture without Graph Neural Networks.
Replaces GCN layers with dense MLPs and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class NoGNNBase(nn.Module):
    """
    Base class replacing GCNBase, using dense operations instead of graph convolutions.
    """
    
    def __init__(self, args, out_fun: str, num_nodes: int):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.positional_features_dims = args.positional_features_dims
        self.embedding_layer_dim = args.embedding_layer_dim
        self.dim_gnn = args.dim_gnn  # Keep naming for compatibility
        
        # Feature transformation layers (replace GCN)
        self.feature_transform = nn.ModuleList()
        
        input_dim = 2 * args.embedding_layer_dim + args.positional_features_dims
        
        # Multiple transformation layers instead of GCN layers
        if args.n_layers_gnn > 0:
            self.feature_transform.append(nn.Linear(input_dim, args.dim_gnn))
            
        for _ in range(args.n_layers_gnn - 1):
            self.feature_transform.append(
                nn.Linear(args.dim_gnn + 2 * args.embedding_layer_dim, args.dim_gnn)
            )
        
        # Global attention mechanism (replace graph message passing)
        if args.n_layers_gnn > 0:
            self.attention = nn.MultiheadAttention(
                embed_dim=args.dim_gnn + 2 * args.embedding_layer_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        
        # Batch normalization layers
        self.bns = nn.ModuleList()
        for _ in range(args.n_layers_gnn):
            self.bns.append(nn.BatchNorm1d(args.dim_gnn + 2 * args.embedding_layer_dim))
        
        # MLP layers (same as original)
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(args.dim_gnn + 2 * args.embedding_layer_dim, args.dim_gnn))
        for _ in range(args.n_layers_nn - 1):
            self.mlp.append(nn.Linear(args.dim_gnn, args.dim_gnn))
        
        self.mlp.append(nn.Linear(args.dim_gnn, args.dim_gnn // 2))
        self.mlp.append(nn.Linear(args.dim_gnn // 2, 1))
        
        # Batch normalization for MLP
        self.bns_mlp = nn.ModuleList()
        for _ in range(args.n_layers_nn):
            self.bns_mlp.append(nn.BatchNorm1d(args.dim_gnn))
        self.bns_mlp.append(nn.BatchNorm1d(args.dim_gnn // 2))
        
        # Output function
        out_fun_selector = {'response': lambda x: x, 'perturbation': lambda x: x}
        self.out_fun = out_fun_selector.get(out_fun, lambda x: x)
    
    def forward(self, x):
        raise NotImplementedError()
    
    def from_node_to_out(self, x1, x2, batch, random_dims):
        """
        Replace graph convolutions with dense transformations and attention.
        """
        # Initial feature concatenation
        x = torch.cat([x1, x2, random_dims], dim=1)
        
        # Apply feature transformations (replace GCN layers)
        for i, (transform, bn) in enumerate(zip(self.feature_transform, self.bns)):
            x_transformed = F.elu(transform(x))
            
            # Reshape for attention: (batch_size, num_nodes, features)
            batch_size = len(torch.unique(batch))
            x_reshaped = x_transformed.view(batch_size, self.num_nodes, -1)
            
            # Apply self-attention (replace graph message passing)
            if hasattr(self, 'attention'):
                x_attended, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
                x_attended = x_attended.view(-1, x_attended.size(-1))
            else:
                x_attended = x_transformed
            
            # Concatenate with original embeddings and apply batch norm
            x = torch.cat([x1, x2, x_attended], dim=1)
            x = bn(x)
        
        # Apply MLP layers (same as original)
        if len(self.mlp) > 1:
            for layer, bn in zip(self.mlp[:-2], self.bns_mlp[:-1]):
                x = bn(F.elu(layer(x)))
            x = self.bns_mlp[-1](F.elu(self.mlp[-2](x)))
        
        return x


class ResponsePredictionModelNoGNN(NoGNNBase):
    """
    Response Prediction Model without GNN layers.
    """
    
    def __init__(self, args, num_nodes: int):
        super().__init__(args, "response", num_nodes)
        
        # Import EmbedLayer from PDGrapher (assuming it's available)
        try:
            from pdgrapher._embed import EmbedLayer
        except ImportError:
            # Fallback: simple embedding layer
            class EmbedLayer(nn.Module):
                def __init__(self, num_vars, num_features, num_categs, hidden_dim):
                    super().__init__()
                    self.embedding = nn.Embedding(num_vars * num_categs, hidden_dim)
                
                def forward(self, x, **kwargs):
                    return self.embedding(x.long()), x
        
        self.num_nodes = num_nodes
        self.embed_layer_pert = EmbedLayer(
            num_nodes, num_features=1, num_categs=2, 
            hidden_dim=args.embedding_layer_dim
        )
        self.embed_layer_ge = EmbedLayer(
            num_nodes, num_features=1, num_categs=500, 
            hidden_dim=args.embedding_layer_dim
        )
        self.positional_embeddings = nn.Embedding(num_nodes, self.positional_features_dims)
        nn.init.normal_(self.positional_embeddings.weight, mean=0.0, std=1.0)
    
    def forward(self, x, batch, topK=None, binarize_intervention=False, 
                mutilate_mutations=None, threshold_input=None):
        x, in_x_binarized = self._get_embeddings(
            x, batch, topK, binarize_intervention, mutilate_mutations, threshold_input
        )
        x = self.mlp[-1](x)
        return self.out_fun(x), in_x_binarized
    
    def _get_embeddings(self, x, batch, topK=None, binarize_intervention=False, 
                       mutilate_mutations=None, threshold_input=None):
        # Positional encodings
        pos_embeddings = self.positional_embeddings(torch.arange(self.num_nodes).to(x.device))
        random_dims = pos_embeddings.repeat(int(x.shape[0] / self.num_nodes), 1)
        
        # Feature embedding
        x_ge, _ = self.embed_layer_ge(
            x[:, 0].view(-1, 1), topK=None, binarize_intervention=False, 
            binarize_input=True, threshold_input=threshold_input
        )
        x_pert, in_x_binarized = self.embed_layer_pert(
            x[:, 1].view(-1, 1), topK=topK, binarize_intervention=binarize_intervention, 
            binarize_input=False, threshold_input=None
        )
        
        # Apply dense transformations instead of graph convolutions
        x = self.from_node_to_out(x_ge, x_pert, batch, random_dims)
        
        return x, in_x_binarized


class PerturbationDiscoveryModelNoGNN(NoGNNBase):
    """
    Perturbation Discovery Model without GNN layers.
    """
    
    def __init__(self, args, num_nodes: int):
        super().__init__(args, "perturbation", num_nodes)
        
        # Import EmbedLayer from PDGrapher (with same fallback)
        try:
            from pdgrapher._embed import EmbedLayer
        except ImportError:
            class EmbedLayer(nn.Module):
                def __init__(self, num_vars, num_features, num_categs, hidden_dim):
                    super().__init__()
                    self.embedding = nn.Embedding(num_vars * num_categs, hidden_dim)
                
                def forward(self, x, **kwargs):
                    return self.embedding(x.long()), x
        
        self.num_nodes = num_nodes
        self.embed_layer_diseased = EmbedLayer(
            num_nodes, num_features=1, num_categs=500, 
            hidden_dim=args.embedding_layer_dim
        )
        self.embed_layer_treated = EmbedLayer(
            num_nodes, num_features=1, num_categs=500, 
            hidden_dim=args.embedding_layer_dim
        )
        self.positional_embeddings = nn.Embedding(num_nodes, self.positional_features_dims)
        nn.init.normal_(self.positional_embeddings.weight, mean=0.0, std=1.0)
    
    def forward(self, x, batch, topK=None, mutilate_mutations=None, threshold_input=None):
        x = self._get_embeddings(x, batch, topK, mutilate_mutations, threshold_input)
        x = self.mlp[-1](x)
        return self.out_fun(x)
    
    def _get_embeddings(self, x, batch, topK=None, mutilate_mutations=None, threshold_input=None):
        # Handle mutations for source detection tasks
        if mutilate_mutations is None:
            mutilate_mutations = torch.zeros(x.shape[0], device=x.device)
        
        # Positional encodings
        pos_embeddings = self.positional_embeddings(torch.arange(self.num_nodes).to(x.device))
        random_dims = pos_embeddings.repeat(int(x.shape[0] / self.num_nodes), 1)
        
        # Feature embedding
        x_diseased, _ = self.embed_layer_diseased(
            x[:, 0].view(-1, 1), topK=None, binarize_input=True, 
            threshold_input=threshold_input["diseased"]
        )
        x_treated, _ = self.embed_layer_treated(
            x[:, 1].view(-1, 1), topK=None, binarize_input=True, 
            threshold_input=threshold_input["treated"]
        )
        
        # Apply dense transformations instead of graph convolutions
        x = self.from_node_to_out(x_diseased, x_treated, batch, random_dims)
        
        return x


# Args dataclass (copy from PDGrapher)
from dataclasses import dataclass
from typing import Dict

@dataclass
class NoGNNArgs:
    """Arguments for NoGNN architecture (same as GCNArgs but without edge_index dependency)."""
    positional_features_dims: int = 16
    embedding_layer_dim: int = 16
    dim_gnn: int = 16  # Keep name for compatibility
    num_vars: int = 1
    n_layers_gnn: int = 1  # Now represents dense layers
    n_layers_nn: int = 2
    
    @classmethod
    def from_dict(cls, args: Dict[str, int]) -> "NoGNNArgs":
        return cls(
            positional_features_dims=args.get("positional_features_dims", 16),
            embedding_layer_dim=args.get("embedding_layer_dim", 16),
            dim_gnn=args.get("dim_gnn", 16),
            num_vars=args.get("num_vars", 1),
            n_layers_gnn=args.get("n_layers_gnn", 1),
            n_layers_nn=args.get("n_layers_nn", 2)
        )


class PDGrapherNoGNN:
    """
    PDGrapher without Graph Neural Networks.
    Maintains the same interface as PDGrapher but uses dense layers and attention.
    """
    
    def __init__(self, num_nodes: int, *, model_kwargs: Dict[str, Any] = {},
                 response_kwargs: Dict[str, Any] = {}, perturbation_kwargs: Dict[str, Any] = {}):
        # Merge kwargs (same logic as original PDGrapher)
        response_kwargs = {**model_kwargs, **response_kwargs}
        perturbation_kwargs = {**model_kwargs, **perturbation_kwargs}
        
        # Pop training flags
        self._train_response_prediction = response_kwargs.pop("train", True)
        self._train_perturbation_discovery = perturbation_kwargs.pop("train", True)
        
        # Create args
        rp_args = NoGNNArgs.from_dict(response_kwargs)
        pd_args = NoGNNArgs.from_dict(perturbation_kwargs)
        
        # Models (no edge_index needed)
        self.response_prediction = ResponsePredictionModelNoGNN(rp_args, num_nodes)
        self.perturbation_discovery = PerturbationDiscoveryModelNoGNN(pd_args, num_nodes)
        
        # Optimizers & Schedulers (same as original)
        self.__optimizer_response_prediction = optim.Adam(
            self.response_prediction.parameters(), lr=0.01
        )
        self.__optimizer_perturbation_discovery = optim.Adam(
            self.perturbation_discovery.parameters(), lr=0.01
        )
        self.__scheduler_response_prediction = lr_scheduler.StepLR(
            self.__optimizer_response_prediction, step_size=350, gamma=0.1
        )
        self.__scheduler_perturbation_discovery = lr_scheduler.StepLR(
            self.__optimizer_perturbation_discovery, step_size=1500, gamma=0.1
        )
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
    
    def get_optimizers_and_schedulers(self):
        """Returns optimizers and schedulers (same interface as PDGrapher)."""
        optimizers = [
            self.__optimizer_response_prediction,
            self.__optimizer_perturbation_discovery
        ]
        schedulers = [
            self.__scheduler_response_prediction,
            self.__scheduler_perturbation_discovery
        ]
        return (optimizers, schedulers)
    
    def set_optimizers_and_schedulers(self, optimizers, schedulers=[None, None]):
        """Set optimizers and schedulers (same interface as PDGrapher)."""
        self.__optimizer_response_prediction = optimizers[0]
        self.__optimizer_perturbation_discovery = optimizers[1]
        self.__scheduler_response_prediction = schedulers[0]
        self.__scheduler_perturbation_discovery = schedulers[1]
