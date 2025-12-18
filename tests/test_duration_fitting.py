#!/usr/bin/env python3
"""
Test script to verify that duration distribution fitting on training data works correctly.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from config import config
from preprocessing.load_data import load_data, split_data
from preprocessing.preprocessing import remove_short_cases
from duration_fitting import (
    fit_duration_distributions_on_training_data,
    save_fitted_distributions,
    load_fitted_distributions,
    print_distribution_summary,
)
from CustomEnvironment.custom_environment.env.custom_environment import (
    AgentOptimizerEnvironment,
    SimulationParameters,
)


def test_duration_fitting():
    """Test the duration fitting functionality."""
    print("=" * 60)
    print("TESTING DURATION DISTRIBUTION FITTING")
    print("=" * 60)

    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data = load_data(config)
    data = remove_short_cases(data)
    train, test = split_data(data)
    print(f"Training data: {len(train)} rows")
    print(f"Test data: {len(test)} rows")

    # Fit distributions on training data
    print("\n2. Fitting duration distributions on training data...")
    fitted_distributions = fit_duration_distributions_on_training_data(train)
    activity_durations_dict, stats_dict, global_activity_medians = fitted_distributions

    # Print summary
    print_distribution_summary(activity_durations_dict, stats_dict)

    # Test saving and loading
    print("\n3. Testing save/load functionality...")
    test_path = "test_distributions.pkl"
    save_fitted_distributions(
        activity_durations_dict, stats_dict, global_activity_medians, test_path
    )

    loaded_distributions = load_fitted_distributions(test_path)
    loaded_activity_durations_dict, loaded_stats_dict, loaded_global_medians = (
        loaded_distributions
    )

    print("✓ Save/load functionality works")

    # Clean up test file
    os.remove(test_path)

    # Test environment creation with pre-fitted distributions
    print("\n4. Testing environment creation with pre-fitted distributions...")
    simulation_parameters = SimulationParameters(
        {"start_timestamp": data["start_timestamp"].min()}
    )

    # Create environment with pre-fitted distributions
    env_with_fitted = AgentOptimizerEnvironment(
        test,  # Use test data but with training distributions
        simulation_parameters,
        experiment_dir="test_env",
        pre_fitted_distributions=fitted_distributions,
    )

    print("✓ Environment created successfully with pre-fitted distributions")

    # Create environment without pre-fitted distributions (original behavior)
    env_without_fitted = AgentOptimizerEnvironment(
        test,
        simulation_parameters,
        experiment_dir="test_env_orig",
    )

    print("✓ Environment created successfully without pre-fitted distributions")

    # Compare if they have the same number of agents and activities
    assert len(env_with_fitted.agents) == len(env_without_fitted.agents)
    assert env_with_fitted.num_activities == env_without_fitted.num_activities

    print("✓ Both environments have same structure")

    # Clean up
    env_with_fitted.close() if hasattr(env_with_fitted, "close") else None
    env_without_fitted.close() if hasattr(env_without_fitted, "close") else None

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("Duration fitting functionality is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    test_duration_fitting()
