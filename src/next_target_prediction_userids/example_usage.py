"""
Example usage of the NextTargetPredictionUserIDs model.

This script demonstrates how to:
1. Initialize the model
2. Create sample training data
3. Train the model with the train_forward method (includes temporal pretraining)
4. Make predictions

To run this example, you'll need PyTorch installed:
    pip install torch
"""

import torch
import torch.optim as optim
from next_target_prediction_userids import NextTargetPredictionUserIDs, NextTargetPredictionBatch


def main():
    """
    Example usage of NextTargetPredictionUserIDs model.
    """
    print("Next Target Prediction using UserIDs - Example Usage")
    print("=" * 60)
    
    # Model configuration
    NUM_USERS = 10000  # Total number of users in the system
    NUM_ACTIONS = 10   # Number of social actions (friend_request, message, etc.)
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    HISTORY_LENGTH = 128
    BATCH_SIZE = 32
    DEVICE = "cpu"
    
    # Initialize the model
    print(f"Initializing model with {NUM_USERS} users and {NUM_ACTIONS} actions...")
    model = NextTargetPredictionUserIDs(
        num_users=NUM_USERS,
        num_actions=NUM_ACTIONS,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_negatives=10,
        dropout=0.1,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create sample training batch
    print("\nCreating sample training batch...")
    
    batch = NextTargetPredictionBatch(
        actor_id=torch.randint(0, NUM_USERS, (model.batch_size,), device=model.device),
        actor_history_actions=torch.randint(0, NUM_ACTIONS, (model.batch_size, HISTORY_LENGTH), device=model.device),
        actor_history_targets=torch.randint(0, NUM_USERS, (model.batch_size, HISTORY_LENGTH), device=model.device),
        actor_history_mask=torch.ones(model.batch_size, HISTORY_LENGTH, device=model.device),
        example_action=torch.randint(0, NUM_ACTIONS, (model.batch_size,), device=model.device),
        example_target=torch.randint(0, NUM_USERS, (model.batch_size,), device=model.device),
    )
    
    # Simulate variable-length histories by masking some entries
    # This demonstrates how to handle users with different numbers of interactions
    # The mask uses 1 for valid entries and 0 for padding
    for i in range(model.batch_size):
        valid_length = torch.randint(20, HISTORY_LENGTH, (1,)).item()
        batch.actor_history_mask[i, valid_length:] = 0  # Mask out padding positions
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        actor_action_repr = model.forward(
            batch.actor_id,
            batch.actor_history_actions,
            batch.actor_history_targets,
            batch.actor_history_mask,
            batch.example_action,
        )
    
    print(f"Actor-action representation shape: {actor_action_repr.shape}")
    print(f"Sample representations: {actor_action_repr[:2]}")
    
    # Training step
    print("\nPerforming training step...")
    model.train()
    
    # Forward pass with training objectives - using pure in-batch negative sampling
    print("Using pure in-batch negative sampling (num_rand_negs=0):")
    results = model.train_forward_with_target(batch, num_rand_negs=0)
    
    print("Training Results (Pure In-Batch Negative Sampling):")
    for key, value in results.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Forward pass with mixed negative sampling
    print("\nUsing mixed negative sampling (num_rand_negs=5):")
    results_mixed = model.train_forward_with_target(batch, num_rand_negs=5)
    
    print("Training Results (Mixed Negative Sampling):")
    for key, value in results_mixed.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Demonstrate how to use this in a training loop
    print("\nDemonstrating training loop with mixed negative sampling...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(3):
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 5
        
        for batch_idx in range(num_batches):
            # Create a new batch (in reality, you'd load from your dataset)
            # Note: In practice, you'd create proper history_masks based on actual user interaction lengths
            batch = NextTargetPredictionBatch(
                actor_id=torch.randint(0, NUM_USERS, (model.batch_size,), device=model.device),
                actor_history_actions=torch.randint(0, NUM_ACTIONS, (model.batch_size, HISTORY_LENGTH), device=model.device),
                actor_history_targets=torch.randint(0, NUM_USERS, (model.batch_size, HISTORY_LENGTH), device=model.device),
                actor_history_mask=torch.ones(model.batch_size, HISTORY_LENGTH, device=model.device),  # All valid for simplicity
                example_action=torch.randint(0, NUM_ACTIONS, (model.batch_size,), device=model.device),
                example_target=torch.randint(0, NUM_USERS, (model.batch_size,), device=model.device),
            )
            
            # Training step with mixed negative sampling
            optimizer.zero_grad()
            results = model.train_forward_with_target(batch, num_rand_negs=3)  # Use 3 additional random negatives
            loss = results['loss']
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += results['accuracy'].item()
        
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")
    
    # Demonstrate inference
    print("\nDemonstrating inference...")
    model.eval()
    
    # Sample inference data
    actor_id = torch.randint(0, NUM_USERS, (2,), device=model.device)  # 2 users
    actor_history_actions = torch.randint(0, NUM_ACTIONS, (2, HISTORY_LENGTH), device=model.device)
    actor_history_targets = torch.randint(0, NUM_USERS, (2, HISTORY_LENGTH), device=model.device)
    actor_history_mask = torch.ones(2, HISTORY_LENGTH, device=model.device)
    action_id = torch.randint(0, NUM_ACTIONS, (2,), device=model.device)
    candidate_targets = torch.randint(0, NUM_USERS, (2, 100), device=model.device)  # 100 candidate friends
    
    with torch.no_grad():
        top_k_scores, top_k_indices = model.predict_top_k(
            actor_id=actor_id,
            actor_history_actions=actor_history_actions,
            actor_history_targets=actor_history_targets,
            actor_history_mask=actor_history_mask,
            action_id=action_id,
            candidate_targets=candidate_targets,
            k=5,  # Top 5 recommendations
        )
    
    print(f"Top-5 recommendation scores shape: {top_k_scores.shape}")
    print(f"Top-5 recommendation indices shape: {top_k_indices.shape}")
    print(f"Top-5 scores for first user: {top_k_scores[0]}")
    
    print("\nExample completed successfully!")
    print("\nIn a real application, you would:")
    print("1. Load user interaction data from your database")
    print("2. Create proper train/validation/test splits")
    print("3. Implement proper data loaders")
    print("4. Add model checkpointing and evaluation metrics")
    print("5. Use the trained model for friend recommendations")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    main() 