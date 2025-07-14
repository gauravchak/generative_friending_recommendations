#!/usr/bin/env python3
"""
Social Tokenized UserIDs (STU) model for friend recommendation.

This model represents users as 3-token sequences and predicts target users
autoregressively, one token at a time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class STUBatch:
    """Batch of STU training examples."""
    actor_stu: torch.Tensor  # [B, 3] - Current user's 3 STU tokens
    actor_history_actions: torch.Tensor  # [B, N] - Historical action IDs
    actor_history_targets: torch.Tensor  # [B, 3N] - Historical target STUs (3 tokens each)
    example_action: torch.Tensor  # [B] - Current action ID
    example_target_stu: torch.Tensor  # [B, 3] - Target user's 3 STU tokens


class NextTargetPredictionSTU(nn.Module):
    """
    Social Tokenized UserIDs model for next target prediction.
    
    Predicts target users as 3-token sequences autoregressively.
    """
    
    def __init__(
        self,
        num_actions: int,
        vocab_size: int = 10000,  # Size of each STU level vocabulary
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        device: str = "cpu"
    ):
        """
        Initialize the STU model.
        
        Args:
            num_actions: Number of possible actions
            vocab_size: Size of vocabulary for each STU level (default: 10000)
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden layers
            device: Device to run on
        """
        super().__init__()
        
        self.num_actions = num_actions
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Embeddings
        self.action_embedding = nn.Embedding(num_actions, embedding_dim, device=device)
        self.stu_embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        
        # User-action representation (shared across all towers)
        self.user_action_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim, device=device),  # actor_stu + history + action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.ReLU()
        )
        
        # Tower-specific layers
        self.tower_1_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size, device=device)
        )
        
        self.tower_2_mlp = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim, device=device),  # + token_0 embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size, device=device)
        )
        
        self.tower_3_mlp = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim * 2, hidden_dim, device=device),  # + token_0, token_1 embeddings
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size, device=device)
        )
    
    def get_user_action_repr(self, actor_stu: torch.Tensor, history_actions: torch.Tensor, 
                           history_targets: torch.Tensor, example_action: torch.Tensor) -> torch.Tensor:
        """
        Get unified user-action representation.
        
        Args:
            actor_stu: [B, 3] - Current user's STU tokens
            history_actions: [B, N] - Historical action IDs
            history_targets: [B, 3N] - Historical target STUs
            example_action: [B] - Current action ID
            
        Returns:
            user_action_repr: [B, hidden_dim] - Unified representation
        """
        batch_size = actor_stu.shape[0]
        
        # 1. Encode actor STU (mean pooling over 3 tokens)
        actor_stu_emb = self.stu_embedding(actor_stu)  # [B, 3, embedding_dim]
        actor_repr = actor_stu_emb.mean(dim=1)  # [B, embedding_dim]
        
        # 2. Encode history (mean pooling over all historical tokens)
        if history_actions.shape[1] > 0:  # If there's history
            # Encode actions
            history_action_emb = self.action_embedding(history_actions)  # [B, N, embedding_dim]
            history_action_repr = history_action_emb.mean(dim=1)  # [B, embedding_dim]
            
            # Encode target STUs (reshape to [B, N, 3, embedding_dim] then mean pool)
            history_targets_reshaped = history_targets.view(batch_size, -1, 3)  # [B, N, 3]
            history_target_emb = self.stu_embedding(history_targets_reshaped)  # [B, N, 3, embedding_dim]
            history_target_repr = history_target_emb.mean(dim=(1, 2))  # [B, embedding_dim]
            
            # Combine action and target representations
            history_repr = (history_action_repr + history_target_repr) / 2  # [B, embedding_dim]
        else:
            # No history - use zero vector
            history_repr = torch.zeros(batch_size, self.embedding_dim, device=self.device)
        
        # 3. Encode example action
        action_repr = self.action_embedding(example_action)  # [B, embedding_dim]
        
        # 4. Combine all representations
        combined_repr = torch.cat([actor_repr, history_repr, action_repr], dim=1)  # [B, 3*embedding_dim]
        
        # 5. Pass through MLP
        user_action_repr = self.user_action_mlp(combined_repr)  # [B, hidden_dim]
        
        return user_action_repr
    
    def tower_1(self, user_action_repr: torch.Tensor) -> torch.Tensor:
        """
        Predict first token of target STU.
        
        Args:
            user_action_repr: [B, hidden_dim] - User-action representation
            
        Returns:
            token_0_probs: [B, vocab_size] - Probability distribution over first token
        """
        return self.tower_1_mlp(user_action_repr)
    
    def tower_2(self, user_action_repr: torch.Tensor, token_0: torch.Tensor) -> torch.Tensor:
        """
        Predict second token of target STU (given first token).
        
        Args:
            user_action_repr: [B, hidden_dim] - User-action representation
            token_0: [B] - First token (for teacher forcing)
            
        Returns:
            token_1_probs: [B, vocab_size] - Probability distribution over second token
        """
        token_0_emb = self.stu_embedding(token_0)  # [B, embedding_dim]
        combined = torch.cat([user_action_repr, token_0_emb], dim=1)  # [B, hidden_dim + embedding_dim]
        return self.tower_2_mlp(combined)
    
    def tower_3(self, user_action_repr: torch.Tensor, token_0: torch.Tensor, token_1: torch.Tensor) -> torch.Tensor:
        """
        Predict third token of target STU (given first two tokens).
        
        Args:
            user_action_repr: [B, hidden_dim] - User-action representation
            token_0: [B] - First token
            token_1: [B] - Second token
            
        Returns:
            token_2_probs: [B, vocab_size] - Probability distribution over third token
        """
        token_0_emb = self.stu_embedding(token_0)  # [B, embedding_dim]
        token_1_emb = self.stu_embedding(token_1)  # [B, embedding_dim]
        combined = torch.cat([user_action_repr, token_0_emb, token_1_emb], dim=1)  # [B, hidden_dim + 2*embedding_dim]
        return self.tower_3_mlp(combined)
    
    def train_forward_with_target(self, batch: STUBatch) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with teacher forcing.
        
        Args:
            batch: STUBatch containing training data
            
        Returns:
            Dictionary with losses and metrics
        """
        # Get user-action representation (shared across all towers)
        user_action_repr = self.get_user_action_repr(
            batch.actor_stu,
            batch.actor_history_actions,
            batch.actor_history_targets,
            batch.example_action
        )
        
        # Extract target tokens for teacher forcing
        target_token_0 = batch.example_target_stu[:, 0]  # [B]
        target_token_1 = batch.example_target_stu[:, 1]  # [B]
        target_token_2 = batch.example_target_stu[:, 2]  # [B]
        
        # Run all 3 towers with teacher forcing
        token_0_logits = self.tower_1(user_action_repr)
        token_1_logits = self.tower_2(user_action_repr, target_token_0)
        token_2_logits = self.tower_3(user_action_repr, target_token_0, target_token_1)
        
        # Compute cross-entropy losses
        loss_0 = F.cross_entropy(token_0_logits, target_token_0)
        loss_1 = F.cross_entropy(token_1_logits, target_token_1)
        loss_2 = F.cross_entropy(token_2_logits, target_token_2)
        
        total_loss = loss_0 + loss_1 + loss_2
        
        # Compute accuracy (argmax prediction)
        token_0_pred = token_0_logits.argmax(dim=1)
        token_1_pred = token_1_logits.argmax(dim=1)
        token_2_pred = token_2_logits.argmax(dim=1)
        
        accuracy_0 = (token_0_pred == target_token_0).float().mean()
        accuracy_1 = (token_1_pred == target_token_1).float().mean()
        accuracy_2 = (token_2_pred == target_token_2).float().mean()
        
        # Overall accuracy (all tokens correct)
        all_correct = (token_0_pred == target_token_0) & (token_1_pred == target_token_1) & (token_2_pred == target_token_2)
        overall_accuracy = all_correct.float().mean()
        
        return {
            'loss': total_loss,
            'loss_0': loss_0,
            'loss_1': loss_1,
            'loss_2': loss_2,
            'accuracy_0': accuracy_0,
            'accuracy_1': accuracy_1,
            'accuracy_2': accuracy_2,
            'overall_accuracy': overall_accuracy
        } 