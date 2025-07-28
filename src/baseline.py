import torch
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
import src.constants as const
import src.utils as utils
import src.validation as validation
import argparse
from typing import List, Dict, Tuple


class BaselineSourceDetector:
    """Base class for baseline source detection methods."""
    
    def __init__(self, method_name: str):
        self.method_name = method_name
    
    def predict_source_probabilities(self, data) -> torch.Tensor:
        """Predict source probabilities for each node in the graph."""
        raise NotImplementedError("Subclasses must implement predict_source_probabilities")
    
    def predict_source(self, data) -> int:
        """Predict the most likely source node."""
        probs = self.predict_source_probabilities(data)
        return probs.argmax().item()


# TODO: NOT RUMOR CENTRALITY! CHATGPT PLACEHOLDER
class RumorCentralityDetector(BaselineSourceDetector):
    """
    Rumor centrality based source detection.
    
    The rumor centrality measures how effectively a node can spread information
    through the network, considering both the network structure and the current
    node states (infection status).
    """
    
    def __init__(self):
        super().__init__("Rumor Centrality")
    
    def predict_source_probabilities(self, data) -> torch.Tensor:
        """
        Predict source probabilities using rumor centrality.
        
        For each node, calculate how well it explains the current infection pattern
        based on network structure and spreading dynamics.
        """
        # Build NetworkX graph
        edge_list = data.edge_index.t().tolist()
        G = nx.from_edgelist(edge_list, create_using=nx.DiGraph)
        
        # Get current infection status (difference from initial state)
        # Assuming x[:, 1] is the current state and x[:, 0] is initial state
        current_state = data.x[:, 1]  # Current activation levels
        initial_state = data.x[:, 0]  # Initial activation levels
        infection_level = current_state - initial_state  # Change from initial
        
        num_nodes = data.x.shape[0]
        centralities = torch.zeros(num_nodes)
        
        for node in range(num_nodes):
            if node not in G.nodes():
                centralities[node] = 0.0
                continue
                
            # Calculate rumor centrality score
            score = self._calculate_rumor_centrality(G, node, infection_level)
            centralities[node] = score
        
        # Normalize to probabilities
        if centralities.sum() > 0:
            centralities = centralities / centralities.sum()
        else:
            # If all scores are 0, uniform distribution
            centralities = torch.ones(num_nodes) / num_nodes
            
        return centralities
    
    def _calculate_rumor_centrality(self, G: nx.DiGraph, candidate_source: int, 
                                  infection_levels: torch.Tensor) -> float:
        """
        Calculate rumor centrality score for a candidate source node.
        
        This measures how well the node explains the observed infection pattern
        based on network distances and spreading potential.
        """
        total_score = 0.0
        
        # Get all nodes that show infection (positive change)
        infected_nodes = torch.where(infection_levels > 0)[0].tolist()
        
        if not infected_nodes:
            return 0.0
        
        for infected_node in infected_nodes:
            if infected_node == candidate_source:
                # Source node gets high score if it's infected
                score = infection_levels[infected_node].item()
            else:
                try:
                    # Calculate shortest path distance
                    distance = nx.shortest_path_length(G, source=candidate_source, 
                                                     target=infected_node)
                    # Score inversely proportional to distance, weighted by infection level
                    score = infection_levels[infected_node].item() / (1 + distance)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # No path or node not in graph
                    score = 0.0
            
            total_score += score
        
        # Normalize by out-degree (spreading potential)
        out_degree = G.out_degree(candidate_source) if candidate_source in G.nodes() else 0
        if out_degree > 0:
            total_score *= np.log(1 + out_degree)  # Logarithmic scaling
        
        return total_score


class RandomDetector(BaselineSourceDetector):
    """Random baseline - uniform probability for all nodes."""
    
    def __init__(self, seed: int = 42):
        super().__init__("Random")
        self.rng = np.random.RandomState(seed)
    
    def predict_source_probabilities(self, data) -> torch.Tensor:
        num_nodes = data.x.shape[0]
        # Random probabilities that sum to 1
        probs = self.rng.random(num_nodes)
        probs = torch.from_numpy(probs).float()
        return probs / probs.sum()


def load_test_data() -> Tuple[List, List]:
    """Load test data for baseline evaluation."""
    # load gat processed data because it has individual files
    test_data_processed = utils.load_processed_data(split="test", model_type="gat")
    test_data_raw = utils.load_raw_test_data()
    return test_data_processed, test_data_raw


def evaluate_baseline_method(detector: BaselineSourceDetector, 
                           processed_data: List, 
                           raw_data: List) -> Dict:
    """
    Evaluate a baseline method using the validation framework.
    
    Args:
        detector: Baseline detector instance
        processed_data: List of processed PyTorch Geometric data
        raw_data: List of raw data instances
        
    Returns:
        Dict: Evaluation metrics
    """
    print(f"Evaluating {detector.method_name}...")
    
    # Generate predictions (per-node probabilities)
    pred_label_set = []
    pred_sources = []
    
    for i, data in enumerate(tqdm(processed_data, desc=f"Making predictions with {detector.method_name}")):
        probs = detector.predict_source_probabilities(data)
        pred_label_set.append(probs)
        pred_sources.append(probs.argmax().item())
    
    true_sources = utils.extract_gat_true_sources(processed_data)

    # Calculate metrics using the validation framework
    metrics = validation.supervised_metrics(
        pred_label_set=pred_label_set,
        true_sources=true_sources,
        pred_sources=pred_sources,
        processed_data=processed_data,
    )
    
    return metrics


def main():
    """Main function to run baseline evaluations."""
    # Komplett über Environment-Variablen oder constants
    methods = ["all"]  # Default oder über eine andere Methode konfigurieren
    
    # Load test data
    print("Loading test data...")
    processed_data, raw_data = load_test_data()
    print(f"Loaded {len(processed_data)} test samples")
    
    # Define available detectors
    detectors = {
        "rumor": RumorCentralityDetector(),
        "random": RandomDetector()
    }
    
    # Select methods to evaluate
    if "all" in methods:
        methods_to_eval = list(detectors.keys())
    else:
        methods_to_eval = methods
    
    # Evaluate each method
    all_results = {}
    
    for method_name in methods_to_eval:
        detector = detectors[method_name]
        print(f"\n{'='*50}")
        print(f"Evaluating {detector.method_name}")
        print(f"{'='*50}")
        
        results = evaluate_baseline_method(detector, processed_data, raw_data)
        all_results[method_name] = results
        
        print(f"\nResults for {detector.method_name}:")
        print("-" * 30)
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
                    

    from datetime import datetime
    import json
    
    for method_name, results in all_results.items():
        if "error" not in results:
            # Create reports directory structure
            report_dir = Path("reports") / f"baseline_{method_name}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%m%d_%H%M")
            filename = f"{const.NETWORK}_{timestamp}.json"
            
            # Wrap results in the expected format with "metrics" key
            output_data = {
                "network": const.NETWORK,
                "metrics": results,
                "method": method_name
            }
            
            with open(report_dir / filename, "w") as f:
                json.dump(output_data, f, indent=4)
            
            print(f"Results for {method_name} saved to {report_dir / filename}")
    
    # Print summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    # Extract key metrics for comparison
    summary_metrics = ["true positive rate", "source in top 3", "auc_roc"]
    
    print(f"{'Method':<20}", end="")
    for metric in summary_metrics:
        if len(metric) > 12:
            print(f"{metric[:12]:<15}", end="")
        else:
            print(f"{metric:<15}", end="")
    print()
    print("-" * 80)
    
    for method_name, results in all_results.items():
        if "error" not in results and "metrics" in results:
            print(f"{method_name:<20}", end="")
            for metric in summary_metrics:
                value = results["metrics"].get(metric, "N/A")
                if isinstance(value, float):
                    print(f"{value:<15.3f}", end="")
                else:
                    print(f"{str(value):<15}", end="")
            print()
        elif "error" in results:
            print(f"{method_name:<20}ERROR: {results['error']}")
    
    print(f"\n{'='*60}")
    print(f"Evaluation completed for network: {const.NETWORK}")
    print(f"Total samples evaluated: {len(processed_data)}")
    print(f"Methods evaluated: {', '.join(methods_to_eval)}")
    print(f"{'='*60}")
    print(f"{metric:<20}", end="")
    print()
    print("-" * (20 + 20 * len(summary_metrics)))
    
    for method_name, results in all_results.items():
        if "error" not in results:
            print(f"{method_name:<20}", end="")
            for metric in summary_metrics:
                value = results.get(metric, "N/A")
                if isinstance(value, (int, float)):
                    print(f"{value:<20.3f}", end="")
                else:
                    print(f"{str(value):<20}", end="")
            print()


if __name__ == "__main__":
    main()