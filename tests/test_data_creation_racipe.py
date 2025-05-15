import pytest
import pandas as pd
from pathlib import Path
from src.data_creation_racipe import (
    get_steady_state_df,
    create_init_conds_file,
    calculate_perturbated_steady_state,
)

def test_get_steady_state_df(tmp_path):
    # Create a mock steady state file
    network_name = "test_network"
    steady_state_file = tmp_path / network_name / "001" / f"{network_name}_steadystate_solutions_001.parquet"
    steady_state_file.parent.mkdir(parents=True, exist_ok=True)
    mock_data = pd.DataFrame({"Time": [1, 2], "State": [0, 1]})
    mock_data.to_parquet(steady_state_file)

    # Test the function
    result = get_steady_state_df(tmp_path, network_name)
    assert result.equals(mock_data)

def test_create_init_conds_file(tmp_path):
    # Create a mock DataFrame
    mock_data = pd.DataFrame({"Node1": [0.1, 0.2], "Node2": [0.3, 0.4]})
    racipe_dir = tmp_path
    network_name = "test_network"

    # Call the function
    create_init_conds_file(mock_data, racipe_dir, network_name)

    # Check if the file was created
    init_conds_file = racipe_dir / "001" / f"{network_name}_init_conds_001.parquet"
    assert init_conds_file.exists()

def test_calculate_perturbated_steady_state():
    # Create a mock steady state DataFrame
    mock_data = pd.DataFrame({
        "Node1": [1.0, 2.0],
        "Node2": [3.0, 4.0],
    })

    # Call the function
    perturbed_state, perturbed_nodes = calculate_perturbated_steady_state(mock_data)

    # Check if the perturbation was applied
    assert not perturbed_state.equals(mock_data)
    assert len(perturbed_nodes) == len(mock_data)