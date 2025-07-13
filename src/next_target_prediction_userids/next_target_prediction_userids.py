import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class NextTargetPredictionBatch:
    """
    Batch data structure for next target prediction using UserIDs.
    
    The history_mask handles variable-length user histories efficiently:
    
    Example: If N=16 (max history length) but a user only has 2 valid interactions:
    - actor_history_actions: [action1, action2, 0, 0, 0, ..., 0]  # padded with zeros
    - actor_history_targets: [user1, user2, 0, 0, 0, ..., 0]     # padded with zeros  
    - actor_history_mask:    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    
    The mask uses 1 for valid entries and 0 for padding, with valid entries typically
    at the end of the sequence (most recent interactions first).
    """
    actor_id: torch.Tensor  # [B] - UserId of the actor who is being recommended friends
    actor_history_actions: torch.Tensor  # [B, N] - list of last N social actions taken by the actor
    actor_history_targets: torch.Tensor  # [B, N] - list of last N UserIds these actions have been taken on
    actor_history_mask: torch.Tensor  # [B, N] - validity mask for variable-length histories
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
        self.user_embeddings = nn.Embedding(num_users, embedding_dim, device=device)
        self.action_embeddings = nn.Embedding(num_actions, embedding_dim, device=device)
        
        # History encoder - processes user's interaction history
        # Each history item is: [action_embedding, target_user_embedding] concatenated
        # So d_model = embedding_dim * 2 (action_dim + user_dim)
        self.history_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim * 2,  # action_embedding + target_user_embedding
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
                device=device,
            ),
            num_layers=2,
        )
        
        # Actor representation network
        self.actor_projection = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim * 2, hidden_dim, device=device),  # actor_id + history_repr
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim, device=device),
        )
        
        # Target prediction network
        self.target_prediction = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim, device=device),  # actor_repr + action_emb + interaction
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim, device=device),  # Output D_emb for dot product operations
        )
        
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
        
        This method handles variable-length histories using a mask. The transformer
        will only attend to valid (masked) positions, and the final representation
        is computed as a mean over valid tokens only.
        
        Data Flow:
        1. Each history item: [action_id, target_user_id] 
        2. Embeddings: [action_embedding(D_emb), target_user_embedding(D_emb)]
        3. Concatenated: [action_features + target_features] = [D_emb * 2]
        4. Transformer processes sequence of these concatenated embeddings
        5. Output: Mean-pooled representation of the entire interaction history
        
        Args:
            history_actions: [B, N] - action IDs (padded with zeros for short histories)
            history_targets: [B, N] - target user IDs (padded with zeros for short histories)
            history_mask: [B, N] - validity mask where 1=valid, 0=padding
                Example: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1] for a user with 2 interactions
                in a sequence of length 16 (valid entries at the end)
            
        Returns:
            torch.Tensor: [B, D_emb * 2] - encoded history representation
        """
        # Get embeddings for each history item
        action_embeds = self.action_embeddings(history_actions)  # [B, N, D_emb] - action embeddings
        target_embeds = self.user_embeddings(history_targets)    # [B, N, D_emb] - target user embeddings
        
        # Concatenate action and target embeddings along the feature dimension
        # Each history item becomes: [action_features, target_user_features]
        # This creates a rich representation of each interaction
        history_embeds = torch.cat([action_embeds, target_embeds], dim=-1)  # [B, N, D_emb * 2]
        
        # Apply transformer encoder with attention mask
        # Convert mask to transformer format (True for tokens to ignore)
        # NOTE: We're not using positional embeddings here - the transformer relies on
        # the order of interactions in the sequence. For better performance, consider
        # adding positional embeddings (e.g., learned positional embeddings or RoPE)
        attention_mask = (history_mask == 0).bool()  # [B, N]
        
        if attention_mask.all():
            # If all tokens are masked, return zeros
            return torch.zeros(history_embeds.size(0), self.embedding_dim * 2, device=history_embeds.device)
        
        # Apply transformer encoder
        encoded_history = self.history_encoder(
            history_embeds,
            src_key_padding_mask=attention_mask
        )  # [B, N, D_emb * 2]
        
        # Mean pooling over valid tokens
        masked_encoded = encoded_history * history_mask.unsqueeze(-1)  # [B, N, D_emb * 2]
        valid_tokens = history_mask.sum(dim=1, keepdim=True)  # [B, 1]
        valid_tokens = torch.clamp(valid_tokens, min=1)  # Avoid division by zero
        
        history_repr = masked_encoded.sum(dim=1) / valid_tokens  # [B, D_emb * 2]
        
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
            actor_history_actions: [B, N] - history of actions (padded with zeros for short histories)
            actor_history_targets: [B, N] - history of target users (padded with zeros for short histories)
            actor_history_mask: [B, N] - validity mask where 1=valid, 0=padding
                Example: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1] for a user with 2 interactions
                in a sequence of length 16 (valid entries at the end)
            example_action: [B] - current action
            
        Returns:
            torch.Tensor: [B, D_emb] - actor-action representation for retrieval
        """
        # Get actor embeddings
        actor_embeds = self.user_embeddings(actor_id)  # [B, D_emb]
        
        # Encode history
        history_repr = self.encode_history(
            actor_history_actions, 
            actor_history_targets, 
            actor_history_mask
        )  # [B, D_emb * 2]
        
        # Combine actor and history representations
        actor_repr = self.actor_projection(
            torch.cat([actor_embeds, history_repr], dim=-1)
        )  # [B, D_emb]
        
        # Get action embeddings
        action_embeds = self.action_embeddings(example_action)  # [B, D_emb]
        
        # Create actor-action representation: concatenate actor_repr, action_repr, and their elementwise product
        actor_action_repr = torch.cat([
            actor_repr, 
            action_embeds, 
            actor_repr * action_embeds  # Elementwise product for interaction modeling
        ], dim=-1)  # [B, D_emb * 3]
        
        # Pass through MLP to get final actor-action representation
        actor_action_repr = self.target_prediction(actor_action_repr)  # [B, D_emb]
        # Don't squeeze - keep the embedding dimension for dot product operations
        # actor_action_repr = actor_action_repr.squeeze(-1)  # [B]
        
        return actor_action_repr
    
    def train_forward_with_target(self, batch: NextTargetPredictionBatch, num_rand_negs: int = 0) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with mixed negative sampling (target prediction only).
        
        This approach combines in-batch negative sampling with additional random negatives:
        1. Uses other examples in the batch as negatives (in-batch negatives)
        2. Optionally adds num_rand_negs additional random negative targets from the entire user space
        3. This provides the benefits of both approaches: realistic negative distribution + diversity
        
        Args:
            batch: NextTargetPredictionBatch containing training data
            num_rand_negs: Number of additional random negative samples to add (default: 0, pure in-batch)
            
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
        )  # [B, D_emb]
        
        # Get target embeddings for all examples in the batch
        target_embeds = self.user_embeddings(batch.example_target)  # [B, D_emb]
        
        # Compute logits: dot product between actor-action representations and target embeddings
        # This gives us [B, B] matrix where logits[i, j] = similarity between example i's actor-action and example j's target
        logits = torch.matmul(actor_action_repr, target_embeds.t())  # [B, B]
        
        # Create mask to handle cases where the same target appears multiple times in the batch
        # mask[i, j] = 0 if example i's target matches example j's target (to avoid treating same target as negative)
        target_mask = (batch.example_target.unsqueeze(1) == batch.example_target.unsqueeze(0)).float()  # [B, B]
        
        if num_rand_negs > 0:
            # Add num_rand_negs additional random negative targets
            random_neg_targets = torch.randint(
                0, self.num_users, 
                (self.batch_size, num_rand_negs), 
                device=self.device
            )  # [B, num_rand_negs]
            
            # Get embeddings for random negative targets
            random_neg_embeds = self.user_embeddings(random_neg_targets)  # [B, num_rand_negs, D_emb]
            
            # Compute logits for random negative examples
            actor_action_repr_expanded = actor_action_repr.unsqueeze(1)  # [B, 1, D_emb]
            random_neg_logits = torch.sum(actor_action_repr_expanded * random_neg_embeds, dim=-1)  # [B, num_rand_negs]
            
            # Concatenate in-batch logits with random negative logits
            all_logits = torch.cat([logits, random_neg_logits], dim=1)  # [B, B + num_rand_negs]
            
            # Extend the mask to cover random negatives
            # For random negatives, we need to check if they match any of the positive targets in the batch
            random_neg_mask = torch.ones(self.batch_size, num_rand_negs, device=self.device)  # [B, num_rand_negs]
            for i in range(self.batch_size):
                # Check if random negative targets match the positive target for this example
                matches = (random_neg_targets[i] == batch.example_target[i]).float()  # [num_rand_negs]
                random_neg_mask[i] = 1 - matches  # [num_rand_negs]
            
            # Concatenate in-batch mask with random negative mask
            all_mask = torch.cat([target_mask, random_neg_mask], dim=1)  # [B, B + num_rand_negs]
            
        else:
            # Pure in-batch negative sampling
            all_logits = logits  # [B, B]
            all_mask = target_mask  # [B, B]
        
        # Apply mask: set logits to large negative value for masked positions
        masked_logits = all_logits - (1 - all_mask) * 1e9  # [B, B + num_rand_negs]
        
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
    
    def temporal_pretraining_loss(
        self, 
        batch: NextTargetPredictionBatch, 
        num_temporal_examples: int = 8
    ) -> Dict[str, torch.Tensor]:
        """
        Temporal pretraining loss using interaction history.
        
        This method creates multiple training examples from a single interaction sequence:
        - Takes history from position 0 to i as "actor history"
        - Uses action at position i+1 as "current action" 
        - Uses target at position i+1 as "positive target"
        - Creates num_temporal_examples such examples per batch item
        
        This provides much more training signal and helps learn temporal patterns.
        
        Args:
            batch: NextTargetPredictionBatch containing training data
            num_temporal_examples: Number of temporal examples to create per batch item
            
        Returns:
            Dict containing temporal loss and metrics
        """
        batch_size = batch.actor_history_actions.size(0)
        history_length = batch.actor_history_actions.size(1)
        
        # Ensure we have enough history for temporal examples
        if history_length < num_temporal_examples + 1:
            num_temporal_examples = max(1, history_length - 1)
        
        # Create temporal examples
        temporal_losses = []
        temporal_accuracies = []
        
        # For each temporal position (last num_temporal_examples positions)
        for i in range(num_temporal_examples):
            # Position in history (from the end)
            pos = history_length - num_temporal_examples + i
            
            # Skip if this position is not valid (masked)
            valid_mask = batch.actor_history_mask[:, pos]  # [B]
            if valid_mask.sum() == 0:
                continue
            
            # Create temporal history (from start to pos-1)
            temporal_history_actions = batch.actor_history_actions[:, :pos]  # [B, pos]
            temporal_history_targets = batch.actor_history_targets[:, :pos]  # [B, pos]
            temporal_history_mask = batch.actor_history_mask[:, :pos]  # [B, pos]
            
            # Current action and target (at position pos)
            current_action = batch.actor_history_actions[:, pos]  # [B]
            current_target = batch.actor_history_targets[:, pos]  # [B]
            
            # Get actor representation from temporal history
            temporal_history_repr = self.encode_history(
                temporal_history_actions,
                temporal_history_targets, 
                temporal_history_mask
            )  # [B, D_emb * 2]
            
            # Get actor embedding (using the same actor_id for all temporal examples)
            actor_embeds = self.user_embeddings(batch.actor_id)  # [B, D_emb]
            
            # Combine actor and temporal history representations
            actor_repr = self.actor_projection(
                torch.cat([actor_embeds, temporal_history_repr], dim=-1)
            )  # [B, D_emb]
            
            # Get action embedding for current action
            action_embeds = self.action_embeddings(current_action)  # [B, D_emb]
            
            # Create actor-action representation
            actor_action_repr = torch.cat([
                actor_repr, 
                action_embeds, 
                actor_repr * action_embeds  # Elementwise product
            ], dim=-1)  # [B, D_emb * 3]
            
            # Final actor-action representation
            actor_action_repr = self.target_prediction(actor_action_repr)  # [B, D_emb]
            
            # Get target embeddings for all users (for negative sampling)
            all_target_embeds = self.user_embeddings.weight  # [num_users, D_emb]
            
            # Compute logits with all possible targets
            logits = torch.matmul(actor_action_repr, all_target_embeds.t())  # [B, num_users]
            
            # Create mask to exclude invalid targets (same as current target)
            target_mask = torch.ones(batch_size, self.num_users, device=self.device)  # [B, num_users]
            for b in range(batch_size):
                if valid_mask[b]:
                    target_mask[b, current_target[b]] = 0  # Exclude positive target
            
            # Apply mask
            masked_logits = logits - (1 - target_mask) * 1e9  # [B, num_users]
            
            # Calculate loss only for valid examples
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) > 0:
                valid_logits = masked_logits[valid_indices]  # [valid_B, num_users]
                valid_targets = current_target[valid_indices]  # [valid_B]
                
                # Cross-entropy loss
                loss = F.cross_entropy(valid_logits, valid_targets)
                temporal_losses.append(loss)
                
                # Accuracy
                predictions = torch.argmax(valid_logits, dim=1)
                accuracy = (predictions == valid_targets).float().mean()
                temporal_accuracies.append(accuracy)
        
        # Aggregate temporal losses
        if temporal_losses:
            total_temporal_loss = torch.stack(temporal_losses).mean()
            avg_temporal_accuracy = torch.stack(temporal_accuracies).mean()
            num_temporal_examples_used = len(temporal_losses)
        else:
            total_temporal_loss = torch.tensor(0.0, device=self.device)
            avg_temporal_accuracy = torch.tensor(0.0, device=self.device)
            num_temporal_examples_used = 0
        
        return {
            'temporal_loss': total_temporal_loss,
            'temporal_accuracy': avg_temporal_accuracy,
            'num_temporal_examples': num_temporal_examples_used,
        }

    def train_forward(
        self, 
        batch: NextTargetPredictionBatch, 
        num_rand_negs: int = 0,
        temporal_weight: float = 0.5,
        num_temporal_examples: int = 8
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with both main loss and temporal pretraining loss.
        
        This combines the standard next-target prediction loss with temporal
        pretraining loss for better learning of sequential patterns.
        
        Args:
            batch: NextTargetPredictionBatch containing training data
            num_rand_negs: Number of additional random negative samples to add
            temporal_weight: Weight for temporal loss (0.0 = no temporal, 1.0 = only temporal)
            num_temporal_examples: Number of temporal examples to create per batch item
            
        Returns:
            Dict containing combined loss and metrics
        """
        # Standard loss
        standard_results = self.train_forward_with_target(batch, num_rand_negs)
        
        # Temporal pretraining loss
        temporal_results = self.temporal_pretraining_loss(batch, num_temporal_examples)
        
        # Combine losses
        combined_loss = (1 - temporal_weight) * standard_results['loss'] + temporal_weight * temporal_results['temporal_loss']
        
        return {
            'loss': combined_loss,
            'standard_loss': standard_results['loss'],
            'temporal_loss': temporal_results['temporal_loss'],
            'accuracy': standard_results['accuracy'],
            'temporal_accuracy': temporal_results['temporal_accuracy'],
            'mean_rank': standard_results['mean_rank'],
            'mrr': standard_results['mrr'],
            'pos_scores_mean': standard_results['pos_scores_mean'],
            'logits_mean': standard_results['logits_mean'],
            'num_negatives': standard_results['num_negatives'],
            'num_temporal_examples': temporal_results['num_temporal_examples'],
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
            actor_history_actions: [B, N] - history of actions (padded with zeros for short histories)
            actor_history_targets: [B, N] - history of target users (padded with zeros for short histories)
            actor_history_mask: [B, N] - validity mask where 1=valid, 0=padding
                Example: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1] for a user with 2 interactions
                in a sequence of length 16 (valid entries at the end)
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
        )  # [B, D_emb]
        
        # Get embeddings for all candidate targets
        candidate_embeds = self.user_embeddings(candidate_targets)  # [B, C, D_emb]
        
        # Compute similarity scores using dot product
        with torch.no_grad():
            scores = torch.sum(
                actor_action_repr.unsqueeze(1) * candidate_embeds, 
                dim=-1
            )  # [B, C]
        
        # Get top-k predictions
        top_k_scores, top_k_indices = torch.topk(scores, k=min(k, candidate_targets.size(1)), dim=1)
        
        return top_k_scores, top_k_indices 