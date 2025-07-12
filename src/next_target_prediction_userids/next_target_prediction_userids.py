import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class NextTargetPredictionBatch:
    """
    Batch data structure for next target prediction using UserIDs.
    """
    actor_id: torch.Tensor  # [B] - UserId of the actor who is being recommended friends
    actor_history_actions: torch.Tensor  # [B, N] - list of last N social actions taken by the actor
    actor_history_targets: torch.Tensor  # [B, N] - list of last N UserIds these actions have been taken on
    actor_history_mask: torch.Tensor  # [B, N] - 0 or 1 depending on validity of that item
    example_action: torch.Tensor  # [B] - action taken on the target of this training example
    example_target: torch.Tensor  # [B] - userId of the target of this training example


class NextTargetPredictionUserIDs(nn.Module):
    """
    PyTorch model for next target prediction using UserIDs.
    This is a stepping stone towards generative recommendations, similar to two tower models.
    """
    
    def __init__(
        self,
        num_users: int,
        num_actions: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_negatives: int = 10,
        dropout: float = 0.1,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """
        Initialize the next target prediction model.
        
        Args:
            num_users: Total number of users in the system
            num_actions: Number of different social actions (friend_request, friend_accept, message, etc.)
            embedding_dim: Dimension of user and action embeddings
            hidden_dim: Hidden dimension for the neural network layers
            num_negatives: Number of negative samples to use for training
            dropout: Dropout rate for regularization
            batch_size: Default batch size for training/inference
            device: Device to use for tensors ("cpu" or "cuda")
        """
        super().__init__()
        
        self.num_users = num_users
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.device = device
        
        # User and action embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim).to(device)
        self.action_embeddings = nn.Embedding(num_actions, embedding_dim).to(device)
        
        # History encoder - processes user's interaction history
        self.history_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim * 2,  # user + action embeddings concatenated
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=2,
        ).to(device)
        
        # Actor representation network
        self.actor_projection = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim * 2, hidden_dim),  # actor_id + history_repr
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)
        
        # Target prediction network
        self.target_prediction = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),  # actor_repr + action_emb + interaction
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),  # Output embedding_dim for dot product operations
        ).to(device)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.action_embeddings.weight)
    
    def encode_history(
        self, 
        history_actions: torch.Tensor, 
        history_targets: torch.Tensor, 
        history_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode user's interaction history using transformer encoder.
        
        Args:
            history_actions: [B, N] - action IDs
            history_targets: [B, N] - target user IDs
            history_mask: [B, N] - validity mask
            
        Returns:
            torch.Tensor: [B, embedding_dim * 2] - encoded history representation
        """
        # Get embeddings
        action_embeds = self.action_embeddings(history_actions)  # [B, N, embedding_dim]
        target_embeds = self.user_embeddings(history_targets)    # [B, N, embedding_dim]
        
        # Concatenate action and target embeddings
        history_embeds = torch.cat([action_embeds, target_embeds], dim=-1)  # [B, N, embedding_dim * 2]
        
        # Apply transformer encoder with attention mask
        # Convert mask to transformer format (True for tokens to ignore)
        attention_mask = (history_mask == 0).bool()  # [B, N]
        
        if attention_mask.all():
            # If all tokens are masked, return zeros
            return torch.zeros(history_embeds.size(0), self.embedding_dim * 2, device=history_embeds.device)
        
        # Apply transformer encoder
        encoded_history = self.history_encoder(
            history_embeds,
            src_key_padding_mask=attention_mask
        )  # [B, N, embedding_dim * 2]
        
        # Mean pooling over valid tokens
        masked_encoded = encoded_history * history_mask.unsqueeze(-1)  # [B, N, embedding_dim * 2]
        valid_tokens = history_mask.sum(dim=1, keepdim=True)  # [B, 1]
        valid_tokens = torch.clamp(valid_tokens, min=1)  # Avoid division by zero
        
        history_repr = masked_encoded.sum(dim=1) / valid_tokens  # [B, embedding_dim * 2]
        
        return history_repr
    
    def forward(
        self, 
        actor_id: torch.Tensor,
        actor_history_actions: torch.Tensor,
        actor_history_targets: torch.Tensor,
        actor_history_mask: torch.Tensor,
        example_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            actor_id: [B] - actor user IDs
            actor_history_actions: [B, N] - history of actions
            actor_history_targets: [B, N] - history of target users
            actor_history_mask: [B, N] - validity mask
            example_action: [B] - current action
            
        Returns:
            torch.Tensor: [B, embedding_dim] - actor-action representation for retrieval
        """
        # Get actor embeddings
        actor_embeds = self.user_embeddings(actor_id)  # [B, embedding_dim]
        
        # Encode history
        history_repr = self.encode_history(
            actor_history_actions, 
            actor_history_targets, 
            actor_history_mask
        )  # [B, embedding_dim * 2]
        
        # Combine actor and history representations
        actor_repr = self.actor_projection(
            torch.cat([actor_embeds, history_repr], dim=-1)
        )  # [B, embedding_dim]
        
        # Get action embeddings
        action_embeds = self.action_embeddings(example_action)  # [B, embedding_dim]
        
        # Create actor-action representation: concatenate actor_repr, action_repr, and their elementwise product
        actor_action_repr = torch.cat([
            actor_repr, 
            action_embeds, 
            actor_repr * action_embeds  # Elementwise product for interaction modeling
        ], dim=-1)  # [B, embedding_dim * 3]
        
        # Pass through MLP to get final actor-action representation
        actor_action_repr = self.target_prediction(actor_action_repr)  # [B, embedding_dim]
        # Don't squeeze - keep the embedding dimension for dot product operations
        # actor_action_repr = actor_action_repr.squeeze(-1)  # [B]
        
        return actor_action_repr
    
    def train_forward(self, batch: NextTargetPredictionBatch, K: int = 0) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with mixed negative sampling.
        
        This approach combines in-batch negative sampling with additional random negatives:
        1. Uses other examples in the batch as negatives (in-batch negatives)
        2. Optionally adds K additional random negative targets from the entire user space
        3. This provides the benefits of both approaches: realistic negative distribution + diversity
        
        Args:
            batch: NextTargetPredictionBatch containing training data
            K: Number of additional random negative samples to add (default: 0, pure in-batch)
            
        Returns:
            Dict containing loss and metrics
        """
        # Get actor-action representations for all examples in the batch
        actor_action_repr = self.forward(
            batch.actor_id,
            batch.actor_history_actions,
            batch.actor_history_targets,
            batch.actor_history_mask,
            batch.example_action,
        )  # [B, embedding_dim]
        
        # Get target embeddings for all examples in the batch
        target_embeds = self.user_embeddings(batch.example_target)  # [B, embedding_dim]
        
        # Compute logits: dot product between actor-action representations and target embeddings
        # This gives us [B, B] matrix where logits[i, j] = similarity between example i's actor-action and example j's target
        logits = torch.matmul(actor_action_repr, target_embeds.t())  # [B, B]
        
        # Create mask to handle cases where the same target appears multiple times in the batch
        # mask[i, j] = 0 if example i's target matches example j's target (to avoid treating same target as negative)
        target_mask = (batch.example_target.unsqueeze(1) == batch.example_target.unsqueeze(0)).float()  # [B, B]
        
        if K > 0:
            # Add K additional random negative targets
            random_neg_targets = torch.randint(
                0, self.num_users, 
                (self.batch_size, K), 
                device=self.device
            )  # [B, K]
            
            # Get embeddings for random negative targets
            random_neg_embeds = self.user_embeddings(random_neg_targets)  # [B, K, embedding_dim]
            
            # Compute logits for random negative examples
            actor_action_repr_expanded = actor_action_repr.unsqueeze(1)  # [B, 1, embedding_dim]
            random_neg_logits = torch.sum(actor_action_repr_expanded * random_neg_embeds, dim=-1)  # [B, K]
            
            # Concatenate in-batch logits with random negative logits
            all_logits = torch.cat([logits, random_neg_logits], dim=1)  # [B, B + K]
            
            # Extend the mask to cover random negatives
            # For random negatives, we need to check if they match any of the positive targets in the batch
            random_neg_mask = torch.ones(self.batch_size, K, device=self.device)  # [B, K]
            for i in range(self.batch_size):
                # Check if random negative targets match the positive target for this example
                matches = (random_neg_targets[i] == batch.example_target[i]).float()  # [K]
                random_neg_mask[i] = 1 - matches  # [K]
            
            # Concatenate in-batch mask with random negative mask
            all_mask = torch.cat([target_mask, random_neg_mask], dim=1)  # [B, B + K]
            
        else:
            # Pure in-batch negative sampling
            all_logits = logits  # [B, B]
            all_mask = target_mask  # [B, B]
        
        # Apply mask: set logits to large negative value for masked positions
        masked_logits = all_logits - (1 - all_mask) * 1e9  # [B, B + K]
        
        # Create labels: diagonal elements are positive examples (each example is positive for itself)
        labels = torch.arange(self.batch_size, device=self.device)  # [B]
        
        # Calculate loss using cross-entropy
        loss = F.cross_entropy(masked_logits, labels)
        
        # Calculate accuracy (how often the positive example has the highest score)
        predictions = torch.argmax(masked_logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        # Calculate ranking metrics
        # For each row, count how many other examples have higher scores than the positive example
        pos_scores = torch.diag(logits)  # [B] - scores for positive examples (from in-batch part)
        pos_scores_expanded = pos_scores.unsqueeze(1)  # [B, 1]
        
        # Count how many other examples have higher scores (excluding the positive example itself)
        rank = (all_logits >= pos_scores_expanded).sum(dim=1).float()  # [B]
        mean_rank = rank.mean()
        mrr = (1.0 / rank).mean()  # Mean Reciprocal Rank
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'mean_rank': mean_rank,
            'mrr': mrr,
            'pos_scores_mean': pos_scores.mean(),
            'logits_mean': all_logits.mean(),
            'num_negatives': torch.tensor(all_logits.size(1) - 1, device=self.device),  # Total number of negatives per example
        }
    
    def predict_top_k(
        self,
        actor_id: torch.Tensor,
        actor_history_actions: torch.Tensor,
        actor_history_targets: torch.Tensor,
        actor_history_mask: torch.Tensor,
        action_id: torch.Tensor,
        candidate_targets: torch.Tensor,
        k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k target users for a given actor and action.
        
        This method computes the actor-action representation and then uses dot product
        with candidate target embeddings to find the most similar targets.
        
        Args:
            actor_id: [B] - actor user IDs
            actor_history_actions: [B, N] - history of actions
            actor_history_targets: [B, N] - history of target users
            actor_history_mask: [B, N] - validity mask
            action_id: [B] - action to predict for
            candidate_targets: [B, C] - candidate target users
            k: number of top predictions to return
            
        Returns:
            Tuple of (top_k_scores, top_k_indices)
        """
        # Get actor-action representation for the given actor and action
        actor_action_repr = self.forward(
            actor_id,
            actor_history_actions,
            actor_history_targets,
            actor_history_mask,
            action_id,
        )  # [B, embedding_dim]
        
        # Get embeddings for all candidate targets
        candidate_embeds = self.user_embeddings(candidate_targets)  # [B, C, embedding_dim]
        
        # Compute similarity scores using dot product
        with torch.no_grad():
            scores = torch.sum(
                actor_action_repr.unsqueeze(1) * candidate_embeds, 
                dim=-1
            )  # [B, C]
        
        # Get top-k predictions
        top_k_scores, top_k_indices = torch.topk(scores, k=min(k, candidate_targets.size(1)), dim=1)
        
        return top_k_scores, top_k_indices 