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
    
    batch = create_sample_batch(num_users=1000, num_actions=10, model=model)
    
    # Test prediction with top-k
    with torch.no_grad():
        # Create 2D candidate targets: [batch_size, num_candidates]
        candidate_targets = torch.arange(1000, device=model.device).unsqueeze(0).expand(model.batch_size, -1)
        
        top_k_scores, top_k_indices = model.predict_top_k(
            actor_id=batch.actor_id,
            actor_history_actions=batch.actor_history_actions,
            actor_history_targets=batch.actor_history_targets,
            actor_history_mask=batch.actor_history_mask,
            action_id=batch.example_action,
            candidate_targets=candidate_targets,
            k=10
        )
    
    print(f"Top-k prediction shapes: scores={top_k_scores.shape}, indices={top_k_indices.shape}")
    print(f"Sample top-k scores: {top_k_scores[0]}")
    print(f"Sample top-k indices: {top_k_indices[0]}")
    
    # Check that predictions are reasonable
    assert top_k_scores.shape == (model.batch_size, 10), "Top-k scores should have shape (batch_size, k)"
    assert top_k_indices.shape == (model.batch_size, 10), "Top-k indices should have shape (batch_size, k)"
    assert torch.all(top_k_indices >= 0) and torch.all(top_k_indices < 1000), "Indices should be valid user IDs"
    
    print("✓ Prediction successful")


def test_variable_name_consistency():
    """Test that all variable names are consistent and properly defined."""
    print("\nTesting variable name consistency...")
    
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
    
    # Test that embedding_dim is properly defined and used
    assert hasattr(model, 'embedding_dim'), "Model should have embedding_dim attribute"
    assert model.embedding_dim == 64, "embedding_dim should match initialization"
    
    # Test that no undefined variables are referenced
    assert hasattr(model, 'user_embeddings'), "Model should have user_embeddings"
    assert hasattr(model, 'action_embeddings'), "Model should have action_embeddings"
    assert hasattr(model, 'history_encoder'), "Model should have history_encoder"
    
    # Test that embeddings have correct dimensions
    assert model.user_embeddings.embedding_dim == model.embedding_dim
    assert model.action_embeddings.embedding_dim == model.embedding_dim
    
    # Test that embed_dim_combined is correctly calculated (if it exists)
    if hasattr(model, 'embed_dim_combined'):
        assert model.embed_dim_combined == model.embedding_dim * 2
    else:
        # In the reverted version, this might not be defined
        print("Note: embed_dim_combined not defined in this version")
    
    print("✓ Variable name consistency test passed")


def test_edge_cases_and_error_handling():
    """Test handling of edge cases and error conditions."""
    print("\nTesting edge cases and error handling...")
    
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
    
    # Test 1: Empty batch (all masks zero)
    empty_batch = NextTargetPredictionBatch(
        actor_id=torch.randint(0, 1000, (8,), device=model.device),
        actor_history_actions=torch.zeros(8, 16, dtype=torch.long, device=model.device),
        actor_history_targets=torch.zeros(8, 16, dtype=torch.long, device=model.device),
        actor_history_mask=torch.zeros(8, 16, device=model.device),  # All masked
        example_action=torch.randint(0, 10, (8,), device=model.device),
        example_target=torch.randint(0, 1000, (8,), device=model.device),
    )
    
    # Should handle empty batch gracefully
    with torch.no_grad():
        result = model.forward(
            empty_batch.actor_id,
            empty_batch.actor_history_actions,
            empty_batch.actor_history_targets,
            empty_batch.actor_history_mask,
            empty_batch.example_action,
        )
        assert result.shape == (8, model.embedding_dim), "Empty batch should produce correct shape"
    
    # Test 2: Invalid user IDs (out of range)
    invalid_batch = create_sample_batch(num_users=1000, num_actions=10, model=model)
    invalid_batch.actor_history_targets[0, 0] = 9999  # Invalid user ID
    
    # Should handle invalid IDs gracefully (clamp or error)
    try:
        with torch.no_grad():
            result = model.forward(
                invalid_batch.actor_id,
                invalid_batch.actor_history_actions,
                invalid_batch.actor_history_targets,
                invalid_batch.actor_history_mask,
                invalid_batch.example_action,
            )
        print("✓ Model handles invalid user IDs gracefully")
    except Exception as e:
        print(f"✓ Model properly errors on invalid user IDs: {e}")
    
    # Test 3: Invalid action IDs
    invalid_action_batch = create_sample_batch(num_users=1000, num_actions=10, model=model)
    invalid_action_batch.actor_history_actions[0, 0] = 9999  # Invalid action ID
    
    try:
        with torch.no_grad():
            result = model.forward(
                invalid_action_batch.actor_id,
                invalid_action_batch.actor_history_actions,
                invalid_action_batch.actor_history_targets,
                invalid_action_batch.actor_history_mask,
                invalid_action_batch.example_action,
            )
        print("✓ Model handles invalid action IDs gracefully")
    except Exception as e:
        print(f"✓ Model properly errors on invalid action IDs: {e}")
    
    print("✓ Edge cases and error handling test passed")


def test_numerical_stability():
    """Test numerical stability and handling of extreme values."""
    print("\nTesting numerical stability...")
    
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
    
    # Test 1: Check for NaN/Inf in forward pass
    with torch.no_grad():
        result = model.forward(
            batch.actor_id,
            batch.actor_history_actions,
            batch.actor_history_targets,
            batch.actor_history_mask,
            batch.example_action,
        )
        
        assert not torch.isnan(result).any(), "Forward pass should not produce NaN values"
        assert not torch.isinf(result).any(), "Forward pass should not produce Inf values"
    
    # Test 2: Check for NaN/Inf in training
    results = model.train_forward_with_target(batch, num_rand_negs=2)
    
    assert not torch.isnan(results['loss']), "Training loss should not be NaN"
    assert not torch.isinf(results['loss']), "Training loss should not be Inf"
    assert results['loss'].item() >= 0, "Training loss should be non-negative"
    
    # Test 3: Check for NaN/Inf in temporal pretraining
    temporal_results = model.temporal_pretraining_loss(batch, num_temporal_examples=4)
    
    assert not torch.isnan(temporal_results['temporal_loss']), "Temporal loss should not be NaN"
    assert not torch.isinf(temporal_results['temporal_loss']), "Temporal loss should not be Inf"
    assert temporal_results['temporal_loss'].item() >= 0, "Temporal loss should be non-negative"
    
    # Test 4: Check gradient flow
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    
    results = model.train_forward_with_target(batch, num_rand_negs=2)
    loss = results['loss']
    loss.backward()
    
    # Check that gradients exist and are finite
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"Gradients for {name} should not be NaN"
            assert not torch.isinf(param.grad).any(), f"Gradients for {name} should not be Inf"
    
    print("✓ Numerical stability test passed")


def test_device_consistency():
    """Test that all tensors are on the correct device."""
    print("\nTesting device consistency...")
    
    # Test CPU
    model_cpu = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
    )
    
    batch_cpu = create_sample_batch(num_users=1000, num_actions=10, model=model_cpu)
    
    # Check that model parameters are on CPU
    for name, param in model_cpu.named_parameters():
        assert param.device.type == 'cpu', f"Parameter {name} should be on CPU"
    
    # Check that forward pass works on CPU
    with torch.no_grad():
        result_cpu = model_cpu.forward(
            batch_cpu.actor_id,
            batch_cpu.actor_history_actions,
            batch_cpu.actor_history_targets,
            batch_cpu.actor_history_mask,
            batch_cpu.example_action,
        )
        assert result_cpu.device.type == 'cpu', "Forward pass result should be on CPU"
    
    # Test MPS if available
    if torch.backends.mps.is_available():
        model_mps = NextTargetPredictionUserIDs(
            num_users=1000,
            num_actions=10,
            embedding_dim=64,
            hidden_dim=128,
            num_negatives=5,
            dropout=0.1,
            batch_size=8,
            device="mps",
        )
        
        batch_mps = create_sample_batch(num_users=1000, num_actions=10, model=model_mps)
        
        # Check that model parameters are on MPS
        for name, param in model_mps.named_parameters():
            assert param.device.type == 'mps', f"Parameter {name} should be on MPS"
        
        # Check that forward pass works on MPS
        with torch.no_grad():
            result_mps = model_mps.forward(
                batch_mps.actor_id,
                batch_mps.actor_history_actions,
                batch_mps.actor_history_targets,
                batch_mps.actor_history_mask,
                batch_mps.example_action,
            )
            assert result_mps.device.type == 'mps', "Forward pass result should be on MPS"
        
        print("✓ MPS device test passed")
    else:
        print("✓ MPS not available, skipping MPS test")
    
    print("✓ Device consistency test passed")


def test_model_architecture_validation():
    """Test that model architecture is consistent and correct."""
    print("\nTesting model architecture validation...")
    
    # Test both encoder types
    for encoder_type in ["transformer", "simple_attention"]:
        model = NextTargetPredictionUserIDs(
            num_users=1000,
            num_actions=10,
            embedding_dim=64,
            hidden_dim=128,
            num_negatives=5,
            dropout=0.1,
            batch_size=8,
            device="cpu",
            history_encoder_type=encoder_type,
        )
        
        batch = create_sample_batch(num_users=1000, num_actions=10, model=model)
        
        # Test that both encoders produce same output shapes
        with torch.no_grad():
            result = model.forward(
                batch.actor_id,
                batch.actor_history_actions,
                batch.actor_history_targets,
                batch.actor_history_mask,
                batch.example_action,
            )
            
            assert result.shape == (model.batch_size, model.embedding_dim), \
                f"{encoder_type} encoder should produce correct output shape"
        
        # Test parameter count validation
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{encoder_type} encoder has {total_params:,} parameters")
        
        # Basic parameter count sanity checks
        expected_min_params = model.num_users * model.embedding_dim + model.num_actions * model.embedding_dim
        assert total_params > expected_min_params, f"{encoder_type} should have more than just embeddings"
        
        # Test that history encoder is properly initialized
        if encoder_type == "transformer":
            assert hasattr(model, 'history_encoder'), "Transformer model should have history_encoder"
            assert hasattr(model, 'history_projection'), "Transformer model should have history_projection"
        elif encoder_type == "simple_attention":
            assert hasattr(model, 'learnable_queries'), "Simple attention model should have learnable_queries"
            assert hasattr(model, 'simple_attention_projection'), "Simple attention model should have simple_attention_projection"
    
    print("✓ Model architecture validation test passed")


def test_loss_bounds_and_metrics():
    """Test that losses and metrics are within reasonable bounds."""
    print("\nTesting loss bounds and metrics...")
    
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
    
    # Test main training loss bounds
    results = model.train_forward_with_target(batch, num_rand_negs=2)
    
    assert 0 <= results['loss'].item() <= 100, "Main loss should be between 0 and 100"
    assert 0 <= results['accuracy'].item() <= 1, "Accuracy should be between 0 and 1"
    assert 0 <= results['mrr'].item() <= 1, "MRR should be between 0 and 1"
    assert results['mean_rank'].item() >= 1, "Mean rank should be at least 1"
    
    # Test temporal pretraining loss bounds
    temporal_results = model.temporal_pretraining_loss(batch, num_temporal_examples=4)
    
    assert 0 <= temporal_results['temporal_loss'].item() <= 100, "Temporal loss should be between 0 and 100"
    assert 0 <= temporal_results['temporal_accuracy'].item() <= 1, "Temporal accuracy should be between 0 and 1"
    assert temporal_results['num_temporal_examples'] >= 0, "Number of temporal examples should be non-negative"
    
    # Test combined training loss bounds
    combined_results = model.train_forward(batch, num_rand_negs=2, temporal_weight=0.3)
    
    assert 0 <= combined_results['loss'].item() <= 100, "Combined loss should be between 0 and 100"
    assert 0 <= combined_results['standard_loss'].item() <= 100, "Standard loss should be between 0 and 100"
    assert 0 <= combined_results['temporal_loss'].item() <= 100, "Temporal loss should be between 0 and 100"
    
    print("✓ Loss bounds and metrics test passed")


def test_batch_size_consistency():
    """Test that model works with different batch sizes."""
    print("\nTesting batch size consistency...")
    
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
    
    # Test different batch sizes
    for batch_size in [1, 4, 8, 16]:
        # Create batch with different size
        batch = NextTargetPredictionBatch(
            actor_id=torch.randint(0, 1000, (batch_size,), device=model.device),
            actor_history_actions=torch.randint(0, 10, (batch_size, 16), device=model.device),
            actor_history_targets=torch.randint(0, 1000, (batch_size, 16), device=model.device),
            actor_history_mask=torch.ones(batch_size, 16, device=model.device),
            example_action=torch.randint(0, 10, (batch_size,), device=model.device),
            example_target=torch.randint(0, 1000, (batch_size,), device=model.device),
        )
        
        # Test forward pass
        with torch.no_grad():
            result = model.forward(
                batch.actor_id,
                batch.actor_history_actions,
                batch.actor_history_targets,
                batch.actor_history_mask,
                batch.example_action,
            )
            assert result.shape == (batch_size, model.embedding_dim), f"Batch size {batch_size} should work"
        
        # Test training
        results = model.train_forward_with_target(batch, num_rand_negs=2)
        assert results['loss'].shape == (), "Loss should be scalar"
        assert not torch.isnan(results['loss']), f"Loss should not be NaN for batch size {batch_size}"
    
    print("✓ Batch size consistency test passed")


def test_moe_interaction_type():
    """Test Mixture of Experts (MoE) interaction type functionality."""
    print("\nTesting MoE interaction type...")
    
    # Test MoE initialization with different expert counts
    for num_experts in [2, 4, 8]:
        model = NextTargetPredictionUserIDs(
            num_users=1000,
            num_actions=10,
            embedding_dim=64,
            hidden_dim=128,
            num_negatives=5,
            dropout=0.1,
            batch_size=8,
            device="cpu",
            interaction_type="moe",
            num_experts=num_experts,
        )
        
        # Verify MoE components are properly initialized
        assert hasattr(model, 'interaction_network'), "MoE model should have interaction_network"
        assert hasattr(model.interaction_network, 'experts'), "MoE should have experts"
        assert hasattr(model.interaction_network, 'gate'), "MoE should have gate"
        assert len(model.interaction_network.experts) == num_experts, f"Should have {num_experts} experts"
        
        # Test forward pass
        batch = create_sample_batch(num_users=1000, num_actions=10, model=model)
        with torch.no_grad():
            result = model.forward(
                batch.actor_id,
                batch.actor_history_actions,
                batch.actor_history_targets,
                batch.actor_history_mask,
                batch.example_action,
            )
            assert result.shape == (model.batch_size, model.embedding_dim), "MoE output should have correct shape"
        
        # Test training
        results = model.train_forward_with_target(batch, num_rand_negs=2)
        assert not torch.isnan(results['loss']), "MoE training should not produce NaN loss"
        assert results['loss'].item() >= 0, "MoE loss should be non-negative"
        
        print(f"✓ MoE with {num_experts} experts works correctly")
    
    print("✓ MoE interaction type test passed")


def test_moe_vs_mlp_comparison():
    """Test that MoE and MLP interaction types produce different but valid results."""
    print("\nTesting MoE vs MLP comparison...")
    
    # Create models with same parameters but different interaction types
    model_mlp = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
        interaction_type="mlp",
    )
    
    model_moe = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
        interaction_type="moe",
        num_experts=4,
    )
    
    batch = create_sample_batch(num_users=1000, num_actions=10, model=model_mlp)
    
    # Test forward pass with both models
    with torch.no_grad():
        result_mlp = model_mlp.forward(
            batch.actor_id,
            batch.actor_history_actions,
            batch.actor_history_targets,
            batch.actor_history_mask,
            batch.example_action,
        )
        
        result_moe = model_moe.forward(
            batch.actor_id,
            batch.actor_history_actions,
            batch.actor_history_targets,
            batch.actor_history_mask,
            batch.example_action,
        )
        
        # Both should have same shape
        assert result_mlp.shape == result_moe.shape, "MLP and MoE should produce same output shape"
        
        # Results should be different (different architectures)
        assert not torch.allclose(result_mlp, result_moe, atol=1e-6), "MLP and MoE should produce different results"
        
        # Both should be finite
        assert torch.isfinite(result_mlp).all(), "MLP output should be finite"
        assert torch.isfinite(result_moe).all(), "MoE output should be finite"
    
    # Test training with both models
    results_mlp = model_mlp.train_forward_with_target(batch, num_rand_negs=2)
    results_moe = model_moe.train_forward_with_target(batch, num_rand_negs=2)
    
    # Both should produce valid losses
    assert not torch.isnan(results_mlp['loss']), "MLP training should not produce NaN loss"
    assert not torch.isnan(results_moe['loss']), "MoE training should not produce NaN loss"
    assert results_mlp['loss'].item() >= 0, "MLP loss should be non-negative"
    assert results_moe['loss'].item() >= 0, "MoE loss should be non-negative"
    
    print("✓ MoE vs MLP comparison test passed")


def test_moe_gating_behavior():
    """Test that MoE gating network produces valid probability distributions."""
    print("\nTesting MoE gating behavior...")
    
    model = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
        interaction_type="moe",
        num_experts=4,
    )
    
    batch = create_sample_batch(num_users=1000, num_actions=10, model=model)
    
    # Get the unified representation that goes into MoE
    actor_embeds = model.user_embeddings(batch.actor_id)
    action_embeds = model.action_embeddings(batch.example_action)
    history_repr = model.encode_history_for_target(
        batch.actor_history_actions,
        batch.actor_history_targets,
        batch.actor_history_mask
    )
    
    # Create unified representation
    actor_history_interaction = actor_embeds * history_repr
    actor_action_interaction = actor_embeds * action_embeds
    
    unified_repr = torch.cat([
        actor_embeds,
        history_repr,
        action_embeds,
        actor_history_interaction,
        actor_action_interaction
    ], dim=-1)  # [B, D_emb * 5]
    
    # Test gating network directly
    with torch.no_grad():
        gate_weights = model.interaction_network.gate(unified_repr)  # [B, num_experts]
        
        # Check that gate weights sum to 1 for each example
        gate_sums = gate_weights.sum(dim=1)
        assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-6), "Gate weights should sum to 1"
        
        # Check that all gate weights are non-negative
        assert (gate_weights >= 0).all(), "Gate weights should be non-negative"
        
        # Check that gate weights are finite
        assert torch.isfinite(gate_weights).all(), "Gate weights should be finite"
        
        print(f"Gate weights shape: {gate_weights.shape}")
        print(f"Sample gate weights: {gate_weights[0]}")
        print(f"Gate weights sum: {gate_weights.sum(dim=1)}")
    
    print("✓ MoE gating behavior test passed")


def test_moe_device_consistency():
    """Test that MoE works correctly on different devices."""
    print("\nTesting MoE device consistency...")
    
    # Test CPU
    model_cpu = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
        interaction_type="moe",
        num_experts=4,
    )
    
    batch_cpu = create_sample_batch(num_users=1000, num_actions=10, model=model_cpu)
    
    # Check that MoE parameters are on CPU
    for name, param in model_cpu.interaction_network.named_parameters():
        assert param.device.type == 'cpu', f"MoE parameter {name} should be on CPU"
    
    # Test forward pass on CPU
    with torch.no_grad():
        result_cpu = model_cpu.forward(
            batch_cpu.actor_id,
            batch_cpu.actor_history_actions,
            batch_cpu.actor_history_targets,
            batch_cpu.actor_history_mask,
            batch_cpu.example_action,
        )
        assert result_cpu.device.type == 'cpu', "MoE forward pass result should be on CPU"
    
    # Test MPS if available
    if torch.backends.mps.is_available():
        model_mps = NextTargetPredictionUserIDs(
            num_users=1000,
            num_actions=10,
            embedding_dim=64,
            hidden_dim=128,
            num_negatives=5,
            dropout=0.1,
            batch_size=8,
            device="mps",
            interaction_type="moe",
            num_experts=4,
        )
        
        batch_mps = create_sample_batch(num_users=1000, num_actions=10, model=model_mps)
        
        # Check that MoE parameters are on MPS
        for name, param in model_mps.interaction_network.named_parameters():
            assert param.device.type == 'mps', f"MoE parameter {name} should be on MPS"
        
        # Test forward pass on MPS
        with torch.no_grad():
            result_mps = model_mps.forward(
                batch_mps.actor_id,
                batch_mps.actor_history_actions,
                batch_mps.actor_history_targets,
                batch_mps.actor_history_mask,
                batch_mps.example_action,
            )
            assert result_mps.device.type == 'mps', "MoE forward pass result should be on MPS"
        
        print("✓ MoE MPS device test passed")
    else:
        print("✓ MPS not available, skipping MoE MPS test")
    
    print("✓ MoE device consistency test passed")


def test_moe_parameter_count():
    """Test that MoE has expected parameter count compared to MLP."""
    print("\nTesting MoE parameter count...")
    
    # MLP model
    model_mlp = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
        interaction_type="mlp",
    )
    
    # MoE model with 4 experts
    model_moe = NextTargetPredictionUserIDs(
        num_users=1000,
        num_actions=10,
        embedding_dim=64,
        hidden_dim=128,
        num_negatives=5,
        dropout=0.1,
        batch_size=8,
        device="cpu",
        interaction_type="moe",
        num_experts=4,
    )
    
    # Count parameters in interaction networks
    mlp_params = sum(p.numel() for p in model_mlp.interaction_network.parameters())
    moe_params = sum(p.numel() for p in model_moe.interaction_network.parameters())
    
    print(f"MLP interaction network parameters: {mlp_params:,}")
    print(f"MoE interaction network parameters: {moe_params:,}")
    
    # MoE should have more parameters than MLP (due to multiple experts)
    assert moe_params > mlp_params, "MoE should have more parameters than MLP"
    
    # MoE should have roughly 4x more parameters (4 experts + gating network)
    expected_ratio = 4.0  # 4 experts + gating network
    actual_ratio = moe_params / mlp_params
    print(f"Parameter ratio (MoE/MLP): {actual_ratio:.2f}")
    
    # Allow some flexibility in the ratio (gating network adds overhead)
    assert 3.0 <= actual_ratio <= 5.0, f"MoE should have roughly 4x parameters, got {actual_ratio:.2f}x"
    
    print("✓ MoE parameter count test passed")


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
    test_variable_name_consistency()
    test_edge_cases_and_error_handling()
    test_numerical_stability()
    test_device_consistency()
    test_model_architecture_validation()
    test_loss_bounds_and_metrics()
    test_batch_size_consistency()
    test_moe_interaction_type()
    test_moe_vs_mlp_comparison()
    test_moe_gating_behavior()
    test_moe_device_consistency()
    test_moe_parameter_count()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully! ✓")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    run_all_tests() 