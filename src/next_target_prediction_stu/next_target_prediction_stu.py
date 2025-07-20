#!/usr/bin/env python3
"""
Social Tokenized UserIDs (STU) model for friend recommendation.

This model represents users as token sequences and predicts target users
autoregressively, one token at a time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class STUBatch:
    """Batch of STU training examples."""
    actor_stu: torch.Tensor  # [B, num_codebooks] - Current user's STU tokens
    actor_history_actions: torch.Tensor  # [B, N] - Historical action IDs
    actor_history_targets: torch.Tensor  # [B, num_codebooks*N] - Historical target STUs
    actor_history_mask: torch.Tensor  # [B, N] - Validity mask for variable-length histories
    example_action: torch.Tensor  # [B] - Current action ID
    example_target_stu: torch.Tensor  # [B, num_codebooks] - Target user's STU tokens


class NextTargetPredictionSTU(nn.Module):
    """
    Social Tokenized UserIDs model for next target prediction.

    Predicts target users as token sequences autoregressively.
    """

    def __init__(
        self,
        num_actions: int,
        num_codebooks: int = 3,  # Number of tokens per user (was hardcoded to 3)
        vocab_size: int = 10000,  # Size of each STU level vocabulary
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        device: str = "cpu"
    ):
        """
        Initialize the STU model.

        Args:
            num_actions: Number of possible actions
            num_codebooks: Number of tokens per user (default: 3)
            vocab_size: Size of vocabulary for each STU level (default: 10000)
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden layers
            device: Device to run on
        """
        super().__init__()

        self.num_actions = num_actions
        self.num_codebooks = num_codebooks
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

        # Unified tower layers - one for each codebook position
        self.tower_layers = nn.ModuleList()
        for i in range(num_codebooks):
            # Input size: hidden_dim + i * embedding_dim (for previous tokens)
            input_size = hidden_dim + i * embedding_dim
            tower = nn.Sequential(
                nn.Linear(input_size, hidden_dim, device=device),
                nn.ReLU(),
                nn.Linear(hidden_dim, vocab_size, device=device)
            )
            self.tower_layers.append(tower)

    def get_user_action_repr(self, actor_stu: torch.Tensor, history_actions: torch.Tensor,
                           history_targets: torch.Tensor, history_mask: torch.Tensor, example_action: torch.Tensor) -> torch.Tensor:
        """
        Get unified user-action representation.

        Args:
            actor_stu: [B, num_codebooks] - Current user's STU tokens
            history_actions: [B, N] - Historical action IDs
            history_targets: [B, num_codebooks*N] - Historical target STUs
            history_mask: [B, N] - Validity mask where 1=valid, 0=padding
            example_action: [B] - Current action ID

        Returns:
            user_action_repr: [B, hidden_dim] - Unified representation
        """
        batch_size = actor_stu.shape[0]

        # 1. Encode actor STU (mean pooling over num_codebooks tokens)
        actor_stu_emb = self.stu_embedding(actor_stu)  # [B, num_codebooks, embedding_dim]
        actor_repr = actor_stu_emb.mean(dim=1)  # [B, embedding_dim]

        # 2. Encode history (mean pooling over all historical tokens)
        if history_actions.shape[1] > 0:  # If there's history
            # Apply mask to actions
            history_action_emb = self.action_embedding(history_actions)  # [B, N, embedding_dim]
            masked_action_emb = history_action_emb * history_mask.unsqueeze(-1)  # [B, N, embedding_dim]
            history_action_repr = masked_action_emb.sum(dim=1) / (history_mask.sum(dim=1, keepdim=True) + 1e-8)  # [B, embedding_dim]

            # Encode target STUs (reshape to [B, N, num_codebooks, embedding_dim] then mean pool)
            history_targets_reshaped = history_targets.view(batch_size, -1, self.num_codebooks)  # [B, N, num_codebooks]
            history_target_emb = self.stu_embedding(history_targets_reshaped)  # [B, N, num_codebooks, embedding_dim]
            masked_target_emb = history_target_emb * history_mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, num_codebooks, embedding_dim]
            history_target_repr = masked_target_emb.sum(dim=(1, 2)) / (history_mask.sum(dim=1, keepdim=True) * self.num_codebooks + 1e-8)  # [B, embedding_dim]

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

    def predict_token(self, user_action_repr: torch.Tensor, previous_tokens: List[torch.Tensor], token_idx: int) -> torch.Tensor:
        """
        Predict token at position token_idx given previous tokens.

        Args:
            user_action_repr: [B, hidden_dim] - User-action representation
            previous_tokens: List of [B] tensors - Previous tokens (empty for first token)
            token_idx: Index of token to predict (0-based)

        Returns:
            token_probs: [B, vocab_size] - Probability distribution over token
        """
        # Build input by concatenating user_action_repr with previous token embeddings
        inputs = [user_action_repr]
        
        for prev_token in previous_tokens:
            prev_token_emb = self.stu_embedding(prev_token)  # [B, embedding_dim]
            inputs.append(prev_token_emb)
        
        combined = torch.cat(inputs, dim=1)  # [B, hidden_dim + num_prev_tokens * embedding_dim]
        
        # Use the appropriate tower layer
        return self.tower_layers[token_idx](combined)

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
            batch.actor_history_mask,
            batch.example_action
        )

        # Extract target tokens for teacher forcing
        target_tokens = [batch.example_target_stu[:, i] for i in range(self.num_codebooks)]  # List of [B] tensors

        # Run all towers with teacher forcing
        token_logits = []
        previous_tokens = []
        
        for i in range(self.num_codebooks):
            logits = self.predict_token(user_action_repr, previous_tokens, i)
            token_logits.append(logits)
            previous_tokens.append(target_tokens[i])  # Use ground truth for teacher forcing

        # Compute cross-entropy losses
        losses = []
        accuracies = []
        
        for i in range(self.num_codebooks):
            loss = F.cross_entropy(token_logits[i], target_tokens[i])
            losses.append(loss)
            
            # Compute accuracy
            pred = token_logits[i].argmax(dim=1)
            accuracy = (pred == target_tokens[i]).float().mean()
            accuracies.append(accuracy)

        total_loss = sum(losses)

        # Overall accuracy (all tokens correct)
        all_correct = torch.ones(batch.actor_stu.shape[0], dtype=torch.bool, device=self.device)
        for i in range(self.num_codebooks):
            pred = token_logits[i].argmax(dim=1)
            all_correct = all_correct & (pred == target_tokens[i])
        overall_accuracy = all_correct.float().mean()

        # Build return dictionary
        result = {
            'loss': total_loss,
            'overall_accuracy': overall_accuracy
        }
        
        # Add individual losses and accuracies
        for i in range(self.num_codebooks):
            result[f'loss_{i}'] = losses[i]
            result[f'accuracy_{i}'] = accuracies[i]

        return result
