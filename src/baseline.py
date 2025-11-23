import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import src.constants as const
import src.utils as utils
import src.validation as validation
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
    

class JordanCenterDetector(BaselineSourceDetector):
    """
    Jordan Center (Rumor Center) based source detection.
    
    The Jordan Center assumes that infection spreads outward from a single source,
    and finds the node with minimal eccentricity in the infected subgraph.
    The source should have minimal maximum distance to all infected nodes.
    """
    
    def __init__(self):
        super().__init__("Jordan Center")
    
    def predict_source_probabilities(self, data) -> torch.Tensor:
        """
        Predict source probabilities using Jordan Center method.
        
        For each node, calculate its eccentricity in the infected subgraph
        and assign higher probability to nodes with lower eccentricity.
        """
        # Build NetworkX graph
        edge_list = data.edge_index.t().tolist()
        G = nx.from_edgelist(edge_list, create_using=nx.DiGraph)
        # Directed for distance calc

        # Get current infection status (difference from initial state)
        current_state = data.x[:, 1]  # Current activation levels
        initial_state = data.x[:, 0]  # Initial activation levels
        infection_level = current_state - initial_state  # Change from initial
        
        # Find infected nodes (nodes with positive change)
        infected_nodes = torch.where(infection_level > 0)[0].tolist()
        
        num_nodes = data.x.shape[0]
        eccentricities = torch.full((num_nodes,), float('inf'))
        
        if len(infected_nodes) == 0:
            # No infected nodes, return uniform distribution
            return torch.ones(num_nodes) / num_nodes
        
        if len(infected_nodes) == 1:
            # Only one infected node, it's the most likely source
            probs = torch.zeros(num_nodes)
            probs[infected_nodes[0]] = 1.0
            return probs
        
        # Calculate eccentricity for each node in the full graph
        for node in G.nodes():
                
            # Calculate maximum distance from this node to all infected nodes
            max_distance = 0
            valid_candidate = True
            
            for infected_node in infected_nodes:
                try:
                    distance = nx.shortest_path_length(G, source=node, target=infected_node)
                    max_distance = max(max_distance, distance)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # Node cannot reach some infected nodes, invalid candidate
                    valid_candidate = False
                    break
            
            if valid_candidate:
                eccentricities[node] = max_distance
        
        # Convert eccentricities to probabilities (lower eccentricity = higher probability)
        # Handle the case where all eccentricities are infinite
        finite_ecc = eccentricities[eccentricities != float('inf')]
        
        if len(finite_ecc) == 0:
            # All nodes have infinite eccentricity, return uniform over infected nodes
            probs = torch.zeros(num_nodes)
            for node in infected_nodes:
                probs[node] = 1.0 / len(infected_nodes)
            return probs
        
        # Invert eccentricities: lower eccentricity = higher score
        min_ecc = finite_ecc.min()
        max_ecc = finite_ecc.max()
        
        probs = torch.zeros(num_nodes)
        
        if min_ecc == max_ecc:
            # All finite eccentricities are the same, uniform over valid nodes
            valid_nodes = torch.where(eccentricities != float('inf'))[0]
            for node in valid_nodes:
                probs[node] = 1.0 / len(valid_nodes)
        else:
            # Convert to probabilities: invert and normalize
            for node in range(num_nodes):
                if eccentricities[node] != float('inf'):
                    # Higher score for lower eccentricity
                    probs[node] = max_ecc - eccentricities[node] + 1
        
        # Normalize to probabilities
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # Fallback to uniform distribution
            probs = torch.ones(num_nodes) / num_nodes
            
        return probs
    

class SmallestSupersetDetector(BaselineSourceDetector):
    """
    Source detection: Knoten, dessen Erreichbarkeitsmenge das kleinste
    Superset der infizierten Knoten ist.
    """

    def __init__(self):
        super().__init__("SmallestSuperset")

    def predict_source_probabilities(self, data) -> torch.Tensor:
        # Graph aufbauen (gerichtet, da Reachability relevant)
        edge_list = data.edge_index.t().tolist()
        G = nx.from_edgelist(edge_list, create_using=nx.DiGraph)

        # Infektionsstatus bestimmen
        current_state = data.x[:, 1]
        initial_state = data.x[:, 0]
        infection_level = current_state - initial_state
        infected_nodes = set(torch.where(infection_level > 0)[0].tolist())

        num_nodes = data.x.shape[0]
        scores = torch.zeros(num_nodes)

        if not infected_nodes:
            # Keine Infektion -> uniform
            return torch.ones(num_nodes) / num_nodes

        # Für jeden Knoten: Reachability und Supersetprüfung
        min_size = float('inf')
        candidates = []

        for node in range(num_nodes):
            if node not in G.nodes():
                continue

            reachable = nx.descendants(G, node) | {node}  # inkl. Startknoten
            if infected_nodes.issubset(reachable):
                size = len(reachable)
                if size < min_size:
                    min_size = size
                    candidates = [node]
                elif size == min_size:
                    candidates.append(node)

        # Wahrscheinlichkeiten setzen
        if candidates:
            for node in candidates:
                scores[node] = 1.0 / len(candidates)
        else:
            # Kein Knoten erreicht alle Infizierten -> fallback
            scores = torch.ones(num_nodes) / num_nodes

        return scores


class OverlapDetector(BaselineSourceDetector):
    """
    Source detection: bewertet Knoten nach Jaccard-Overlap zwischen
    erreichbaren Knoten und infizierten Knoten.
    """

    def __init__(self):
        super().__init__("Overlap")

    def predict_source_probabilities(self, data) -> torch.Tensor:
        # Graph aufbauen (gerichtet)
        edge_list = data.edge_index.t().tolist()
        G = nx.from_edgelist(edge_list, create_using=nx.DiGraph)

        # Infektionsstatus bestimmen
        current_state = data.x[:, 1]
        initial_state = data.x[:, 0]
        infection_level = current_state - initial_state
        infected_nodes = set(torch.where(infection_level > 0)[0].tolist())

        num_nodes = data.x.shape[0]
        scores = torch.zeros(num_nodes)

        if not infected_nodes:
            # Keine Infektion -> uniform
            return torch.ones(num_nodes) / num_nodes

        # Für jeden Knoten: Overlap-Score berechnen
        for node in range(num_nodes):
            if node not in G.nodes():
                continue

            reachable = nx.descendants(G, node) | {node}  # inkl. Startknoten
            intersection_size = len(reachable & infected_nodes)
            union_size = len(reachable | infected_nodes)

            if union_size > 0:
                scores[node] = intersection_size / union_size
            else:
                scores[node] = 0.0

        # Normieren zu Wahrscheinlichkeiten
        if scores.sum() > 0:
            scores = scores / scores.sum()
        else:
            scores = torch.ones(num_nodes) / num_nodes

        return scores



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
    

class AdvancedRandomDetector(BaselineSourceDetector):
    """Random baseline - uniform probability for all nodes."""
    
    def __init__(self, seed: int = 42):
        super().__init__("Random")
        self.rng = np.random.RandomState(seed)
    
    def predict_source_probabilities(self, data) -> torch.Tensor:
        num_nodes = data.x.shape[0]
        # Build NetworkX graph to check out-degrees
        edge_list = data.edge_index.t().tolist()
        G = nx.from_edgelist(edge_list, create_using=nx.DiGraph)
        out_degrees = np.array([G.out_degree(n) if n in G.nodes else 0 for n in range(num_nodes)])
        # Mask: 1 if node has outgoing edges, 0 otherwise
        mask = (out_degrees > 0).astype(float)
        if mask.sum() == 0:
            # If all nodes have out-degree 0, fall back to uniform
            probs = np.ones(num_nodes)
        else:
            # Random values only for nodes with out-degree > 0
            probs = self.rng.random(num_nodes) * mask
            # If all random values are zero (shouldn't happen), fallback
            if probs.sum() == 0:
                probs = mask
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
        "jordan": JordanCenterDetector(),
        "smallestSuperset": SmallestSupersetDetector(),
        "overlap": OverlapDetector(),
        "random": RandomDetector(),
        "advanced_random": AdvancedRandomDetector(),
    }
    
    # Select methods to evaluate
    if "all" in methods:
        methods_to_eval = list(detectors.keys())
    else:
        # remove jordan center from the list
        methods_to_eval = methods
    
    methods_to_eval.remove("jordan")
        
    
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

    
    for method_name, results in all_results.items():
        if "error" not in results:
            method = f"baseline_{method_name}"

            output_data = {
                "network": const.NETWORK,
                "metrics": results,
                "model_type": method_name
            }

            utils.save_metrics(output_data, method_name=method)

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