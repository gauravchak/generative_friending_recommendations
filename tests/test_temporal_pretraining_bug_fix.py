#!/usr/bin/env python3
"""
Test to verify that the temporal pretraining bug is fixed.

This test ensures that:
1. Temporal pretraining loss decreases during training
2. The model can learn from temporal examples
3. No incorrect masking is applied to positive targets
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add the parent directory to the path to import the model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'next_target_prediction_userids'))
from next_target_prediction_userids import NextTargetPredictionUserIDs, NextTargetPredictionBatch

# Import torch functions explicitly to avoid linter errors
from torch import randint, ones


def create_temporal_test_batch(batch_size=4, num_users=100, num_actions=5, history_length=8):
    """Create a test batch with temporal examples."""
    # Create a batch where each example has a sequence of actions
    batch = NextTargetPredictionBatch(
        actor_id=randint(0, num_users, (batch_size,)),
        actor_history_actions=randint(0, num_actions, (batch_size, history_length)),
        actor_history_targets=randint(0, num_users, (batch_size, history_length)),
        actor_history_mask=ones(batch_size, history_length),  # All valid
        example_action=randint(0, num_actions, (batch_size,)),
        example_target=randint(0, num_users, (batch_size,))
    )
    return batch


def test_temporal_pretraining_loss_decreases():
    """Test that temporal pretraining loss decreases during training."""
    print("Testing temporal pretraining loss decreases...")
    
    # Create model
    model = NextTargetPredictionUserIDs(
        num_users=100,
        num_actions=5,
        embedding_dim=32,
        hidden_dim=64,
        device="cpu"
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Create test batch
    batch = create_temporal_test_batch()
    
    # Track losses
    initial_loss = None
    losses = []
    
    # Train for a few steps
    for step in range(10):
        optimizer.zero_grad()
        
        # Get temporal loss
        results = model.temporal_pretraining_loss(batch, num_temporal_examples=4)
        loss = results['temporal_loss']
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Step {step + 1}: Temporal Loss = {loss.item():.4f}, "
              f"Accuracy = {results['temporal_accuracy'].item():.4f}")
    
    # Check that loss decreased
    final_loss = losses[-1]
    loss_decreased = final_loss < initial_loss
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss decreased: {loss_decreased}")
    
    assert loss_decreased, f"Temporal loss should decrease during training. Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    print("✅ Temporal pretraining loss decreases correctly")


def test_no_incorrect_masking():
    """Test that no incorrect masking is applied to positive targets."""
    print("\nTesting no incorrect masking...")
    
    model = NextTargetPredictionUserIDs(
        num_users=100,
        num_actions=5,
        embedding_dim=32,
        hidden_dim=64,
        device="cpu"
    )
    
    batch = create_temporal_test_batch()
    
    # Get temporal loss
    results = model.temporal_pretraining_loss(batch, num_temporal_examples=4)
    
    # Check that we have valid temporal examples
    num_temporal_examples = results['num_temporal_examples']
    assert num_temporal_examples > 0, "Should have valid temporal examples"
    
    print(f"Number of temporal examples: {num_temporal_examples}")
    print(f"Temporal accuracy: {results['temporal_accuracy'].item():.4f}")
    
    # The key test: if masking was incorrect, the loss would be very high
    # and accuracy would be very low because the model couldn't learn
    temporal_loss = results['temporal_loss'].item()
    temporal_accuracy = results['temporal_accuracy'].item()
    
    # Loss should be reasonable (not extremely high due to incorrect masking)
    assert temporal_loss < 10.0, f"Temporal loss should be reasonable, got {temporal_loss}"
    
    print(f"Temporal loss: {temporal_loss:.4f}")
    print("✅ No incorrect masking detected")


def test_combined_training_works():
    """Test that combined training (standard + temporal) works correctly."""
    print("\nTesting combined training...")
    
    model = NextTargetPredictionUserIDs(
        num_users=100,
        num_actions=5,
        embedding_dim=32,
        hidden_dim=64,
        device="cpu"
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    batch = create_temporal_test_batch()
    
    # Test combined training
    results = model.train_forward(
        batch, 
        num_rand_negs=2, 
        temporal_weight=0.5, 
        num_temporal_examples=4
    )
    
    # Check that both losses are present
    assert 'standard_loss' in results, "Should have standard loss"
    assert 'temporal_loss' in results, "Should have temporal loss"
    assert 'loss' in results, "Should have combined loss"
    
    standard_loss = results['standard_loss'].item()
    temporal_loss = results['temporal_loss'].item()
    combined_loss = results['loss'].item()
    
    print(f"Standard loss: {standard_loss:.4f}")
    print(f"Temporal loss: {temporal_loss:.4f}")
    print(f"Combined loss: {combined_loss:.4f}")
    
    # Combined loss should be reasonable
    assert combined_loss < 10.0, f"Combined loss should be reasonable, got {combined_loss}"
    
    # Test that gradients flow
    optimizer.zero_grad()
    combined_loss_tensor = results['loss']
    combined_loss_tensor.backward()
    
    # Check that gradients exist
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_gradients, "Gradients should flow through the model"
    
    print("✅ Combined training works correctly")


def main():
    """Run all tests."""
    print("Testing Temporal Pretraining Bug Fix")
    print("=" * 40)
    
    try:
        test_temporal_pretraining_loss_decreases()
        test_no_incorrect_masking()
        test_combined_training_works()
        
        print("\n" + "=" * 40)
        print("✅ All tests passed! The temporal pretraining bug is fixed.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 