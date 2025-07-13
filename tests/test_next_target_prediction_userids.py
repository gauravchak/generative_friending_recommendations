"""
Test file for Next Target Prediction using UserIDs model.

This file demonstrates how to use the NextTargetPredictionUserIDs model
with sample data and shows the expected behavior.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
import sys
import os

# Add project root to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our model
from src.next_target_prediction_userids.next_target_prediction_userids import NextTargetPredictionUserIDs, NextTargetPredictionBatch


def create_sample_batch(
    num_users: int = 1000,
    num_actions: int = 10,
    history_length: int = 128,
    model=None,
) -> NextTargetPredictionBatch:
    """
    Create a sample batch for testing.
    
    Args:
        num_users: Total number of users in the system
        num_actions: Total number of actions
        history_length: Length of user history
        model: The model instance to get batch_size and device
        
    Returns:
        NextTargetPredictionBatch: Sample batch for testing
    """
    batch_size = model.batch_size if model is not None else 4
    device = model.device if model is not None else "cpu"
    # Create sample actor IDs
    actor_id = torch.randint(0, num_users, (batch_size,), device=device)
    
    # Create sample history - some users have shorter histories
    actor_history_actions = torch.randint(0, num_actions, (batch_size, history_length), device=device)
    actor_history_targets = torch.randint(0, num_users, (batch_size, history_length), device=device)
    
    # Create masks - simulate variable length histories
    actor_history_mask = torch.ones(batch_size, history_length, device=device)
    for i in range(batch_size):
        # Random history length between 10 and history_length
        valid_length = torch.randint(10, history_length, (1,)).item()
        actor_history_mask[i, valid_length:] = 0
    
    # Create example actions and targets
    example_action = torch.randint(0, num_actions, (batch_size,), device=device)
    example_target = torch.randint(0, num_users, (batch_size,), device=device)
    
    return NextTargetPredictionBatch(
        actor_id=actor_id,
        actor_history_actions=actor_history_actions,
        actor_history_targets=actor_history_targets,
        actor_history_mask=actor_history_mask,
        example_action=example_action,
        example_target=example_target,
    )


def create_sample_batch_with_long_history(num_users: int = 1000, num_actions: int = 10, history_length: int = 32, model=None, min_valid: int = 12) -> NextTargetPredictionBatch:
    """
    Create a sample batch where each user has at least `min_valid` valid history entries (for temporal pretraining tests).
    """
    batch = NextTargetPredictionBatch(
        actor_id=torch.randint(0, num_users, (model.batch_size,), device=model.device),
        actor_history_actions=torch.randint(0, num_actions, (model.batch_size, history_length), device=model.device),
        actor_history_targets=torch.randint(0, num_users, (model.batch_size, history_length), device=model.device),
        actor_history_mask=torch.zeros(model.batch_size, history_length, device=model.device),
        example_action=torch.randint(0, num_actions, (model.batch_size,), device=model.device),
        example_target=torch.randint(0, num_users, (model.batch_size,), device=model.device),
    )
    # Guarantee at least min_valid valid entries at the end for each user
    for i in range(model.batch_size):
        batch.actor_history_mask[i, -min_valid:] = 1
    return batch


def test_model_initialization():
    """Test that the model can be initialized properly."""
    print("Testing model initialization...")
    
    model = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print("✓ Model initialization successful")


def test_forward_pass():
    """Test that the model can perform forward pass."""
    print("\nTesting forward pass...")
    
    model = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
    )
    
    batch = create_sample_batch(num_users=1000, num_actions=10, model=model)
    
    # Test forward pass - now returns actor-action representation
    with torch.no_grad():
        actor_action_repr = model.forward(
            batch.actor_id,
            batch.actor_history_actions,
            batch.actor_history_targets,
            batch.actor_history_mask,
            batch.example_action,
        )
    
    print(f"Forward pass output shape: {actor_action_repr.shape}")
    print(f"Sample actor-action representations: {actor_action_repr[:3]}")
    print("✓ Forward pass successful")


def test_train_forward_with_target():
    """Test the training forward pass with different negative sampling strategies."""
    print("\nTesting train_forward_with_target (mixed negative sampling)...")
    
    model = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
    )
    
    batch = create_sample_batch(num_users=1000, num_actions=10, model=model)
    
    # Test pure in-batch negative sampling (num_rand_negs=0)
    print("Testing pure in-batch negative sampling (num_rand_negs=0):")
    results = model.train_forward_with_target(batch, num_rand_negs=0)
    
    print("Training metrics (Pure In-Batch Negative Sampling):")
    for key, value in results.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Check that loss is a scalar
    assert results['loss'].dim() == 0, "Loss should be a scalar"
    assert results['loss'].item() >= 0, "Loss should be non-negative"
    assert results['num_negatives'].item() == 7, "Should have 7 in-batch negatives (batch_size - 1)"
    
    # Test mixed negative sampling (num_rand_negs=3)
    print("\nTesting mixed negative sampling (num_rand_negs=3):")
    results_mixed = model.train_forward_with_target(batch, num_rand_negs=3)
    
    print("Training metrics (Mixed Negative Sampling):")
    for key, value in results_mixed.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Check that loss is a scalar
    assert results_mixed['loss'].dim() == 0, "Loss should be a scalar"
    assert results_mixed['loss'].item() >= 0, "Loss should be non-negative"
    assert results_mixed['num_negatives'].item() == 10, "Should have 10 total negatives (7 in-batch + 3 random)"
    
    print("✓ train_forward_with_target successful")


def test_temporal_pretraining():
    """
    Test the temporal pretraining loss functionality.
    Assumption: The batch must have at least K+1 valid history positions for temporal pretraining to work.
    """
    print("\nTesting temporal pretraining loss...")
    
    model = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
    )
    
    # Ensure at least 8+1 valid history positions for each user
    batch = create_sample_batch_with_long_history(num_users=1000, num_actions=10, history_length=16, model=model, min_valid=9)
    
    # Test temporal pretraining loss alone
    print("Testing temporal pretraining loss:")
    temporal_results = model.temporal_pretraining_loss(batch, num_temporal_examples=8)
    
    print("Temporal pretraining metrics:")
    for key, value in temporal_results.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Check that loss is a scalar and non-negative
    assert temporal_results['temporal_loss'].dim() == 0, "Temporal loss should be a scalar"
    assert temporal_results['temporal_loss'].item() >= 0, "Temporal loss should be non-negative"
    assert temporal_results['num_temporal_examples'] > 0, "Should have some temporal examples"
    
    # Test combined training
    print("\nTesting combined training (standard + temporal):")
    combined_results = model.train_forward(
        batch, 
        num_rand_negs=2, 
        temporal_weight=0.3,
        num_temporal_examples=8
    )
    
    print("Combined training metrics:")
    for key, value in combined_results.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Check that combined loss is reasonable
    assert combined_results['loss'].dim() == 0, "Combined loss should be a scalar"
    assert combined_results['loss'].item() >= 0, "Combined loss should be non-negative"
    assert combined_results['standard_loss'].item() >= 0, "Standard loss should be non-negative"
    assert combined_results['temporal_loss'].item() >= 0, "Temporal loss should be non-negative"
    
    print("✓ Temporal pretraining successful")


def test_training_loop():
    """Test a simple training loop to ensure gradients flow properly."""
    print("\nTesting training loop...")
    
    model = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with mixed negative sampling
    initial_loss = None
    for step in range(5):
        batch = create_sample_batch(num_users=1000, num_actions=10, model=model)
        
        optimizer.zero_grad()
        results = model.train_forward_with_target(batch, num_rand_negs=2)  # Use 2 additional random negatives
        loss = results['loss']
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        print(f"Step {step + 1}: Loss = {loss.item():.4f}, Accuracy = {results['accuracy'].item():.4f}, Negatives = {results['num_negatives'].item()}")
    
    print(f"Initial loss: {initial_loss:.4f}")
    print("✓ Training loop successful")


def test_training_loop_with_temporal():
    """Test a training loop with temporal pretraining."""
    print("\nTesting training loop with temporal pretraining...")
    
    model = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with temporal pretraining
    for step in range(3):
        batch = create_sample_batch(num_users=1000, num_actions=10, model=model)
        
        optimizer.zero_grad()
        results = model.train_forward(
            batch, 
            num_rand_negs=1, 
            temporal_weight=0.4,
            num_temporal_examples=3
        )
        loss = results['loss']
        
        loss.backward()
        optimizer.step()
        
        print(f"Step {step + 1}: Combined Loss = {loss.item():.4f}, "
              f"Standard Loss = {results['standard_loss'].item():.4f}, "
              f"Temporal Loss = {results['temporal_loss'].item():.4f}, "
              f"Temporal Examples = {results['num_temporal_examples']}")
    
    print("✓ Training loop with temporal pretraining successful")


def test_simple_attention():
    """Test the simple attention history encoder."""
    print("\nTesting simple attention history encoder...")
    
    # Test with simple attention
    model_simple = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
        history_encoder_type="simple_attention",
    )
    
    # Test with transformer (default)
    model_transformer = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
        history_encoder_type="transformer",
    )
    
    batch = create_sample_batch(num_users=1000, num_actions=10, model=model_simple)
    
    # Test forward pass with simple attention
    with torch.no_grad():
        simple_output = model_simple.forward(
            batch.actor_id,
            batch.actor_history_actions,
            batch.actor_history_targets,
            batch.actor_history_mask,
            batch.example_action,
        )
        
        transformer_output = model_transformer.forward(
            batch.actor_id,
            batch.actor_history_actions,
            batch.actor_history_targets,
            batch.actor_history_mask,
            batch.example_action,
        )
    
    print(f"Simple attention output shape: {simple_output.shape}")
    print(f"Transformer output shape: {transformer_output.shape}")
    
    # Check that outputs have the same shape
    assert simple_output.shape == transformer_output.shape, "Output shapes should match"
    
    # Test training with simple attention
    results_simple = model_simple.train_forward_with_target(batch, num_rand_negs=0)
    results_transformer = model_transformer.train_forward_with_target(batch, num_rand_negs=0)
    
    print("Simple attention training metrics:")
    for key, value in results_simple.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("Transformer training metrics:")
    for key, value in results_transformer.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Check that both models produce valid results
    assert results_simple['loss'].dim() == 0, "Simple attention loss should be a scalar"
    assert results_transformer['loss'].dim() == 0, "Transformer loss should be a scalar"
    assert results_simple['loss'].item() >= 0, "Simple attention loss should be non-negative"
    assert results_transformer['loss'].item() >= 0, "Transformer loss should be non-negative"
    
    print("✓ Simple attention test successful")


def test_prediction():
    """Test the prediction functionality."""
    print("\nTesting prediction...")
    
    model = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
    )
    
    # Create sample input for prediction
    batch_size = model.batch_size
    num_candidates = 100
    
    actor_id = torch.randint(0, 1000, (batch_size,), device=model.device)
    actor_history_actions = torch.randint(0, 10, (batch_size, 128), device=model.device)
    actor_history_targets = torch.randint(0, 1000, (batch_size, 128), device=model.device)
    actor_history_mask = torch.ones(batch_size, 128, device=model.device)
    action_id = torch.randint(0, 10, (batch_size,), device=model.device)
    candidate_targets = torch.randint(0, 1000, (batch_size, num_candidates), device=model.device)
    
    # Test prediction
    with torch.no_grad():
        top_k_scores, top_k_indices = model.predict_top_k(
            actor_id=actor_id,
            actor_history_actions=actor_history_actions,
            actor_history_targets=actor_history_targets,
            actor_history_mask=actor_history_mask,
            action_id=action_id,
            candidate_targets=candidate_targets,
            k=10,
        )
    
    print(f"Top-k scores shape: {top_k_scores.shape}")
    print(f"Top-k indices shape: {top_k_indices.shape}")
    print(f"Sample top-3 scores for first example: {top_k_scores[0, :3]}")
    print("✓ Prediction successful")


def run_all_tests():
    """Run all tests."""
    print("Running Next Target Prediction UserIDs Tests")
    print("=" * 50)
    
    test_model_initialization()
    test_forward_pass()
    test_train_forward_with_target()
    test_temporal_pretraining()
    test_training_loop()
    test_training_loop_with_temporal()
    test_simple_attention()
    test_prediction()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully! ✓")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    run_all_tests() 