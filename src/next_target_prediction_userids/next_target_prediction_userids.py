import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


class SwiGLU(nn.Module):
    """
    SwiGLU activation function: Swish(xW + b) ⊗ (xV + c)
    
    This is a modern activation function that combines Swish with a gating mechanism.
    It's particularly effective for transformer-based models and can provide better
    feature selection and interaction modeling than GELU.
    """
    def __init__(self, input_dim: int, hidden_dim: int, device: str = "cpu"):
        super().__init__()
        self.w_gate = nn.Linear(input_dim, hidden_dim, device=device)
        self.w_value = nn.Linear(input_dim, hidden_dim, device=device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))  # Swish/SiLU activation
        value = self.w_value(x)
        return gate * value  # Element-wise multiplication


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
        history_encoder_type: str = "transformer",
    ):
        """
        Initialize the next target prediction model.
        
        Args:
            num_users: Total number of users in the system
            num_actions: Number of different social actions 
            (friend_request, friend_accept, message, etc.)
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
        self.history_encoder_type = history_encoder_type
        
        # User and action embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim, device=device)
        self.action_embeddings = nn.Embedding(num_actions, embedding_dim, device=device)
        
        # History encoder - processes user's interaction history
        # Each history item is: [action_embedding, target_user_embedding] concatenated
        # So d_model = embedding_dim * 2 (action_dim + user_dim)
        
        if self.history_encoder_type == "transformer":
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
            
            # History projection layer - reduces D_emb * 2 to D_emb
            self.history_projection = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim, device=device),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif self.history_encoder_type == "simple_attention":
            # Simple attention components (for history_encoder_type="simple_attention")
            self.num_attention_heads = 2  # K=2 learnable query vectors
            self.learnable_queries = nn.Parameter(torch.randn(self.num_attention_heads, embedding_dim * 2, device=device))
            self.simple_attention_projection = nn.Sequential(
                nn.Linear(self.num_attention_heads * embedding_dim * 2, embedding_dim, device=device),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unknown history_encoder_type: {self.history_encoder_type}")
        
        # Actor representation network
        self.actor_projection = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim + embedding_dim, hidden_dim, device=device),  # actor_id + history_repr + latent_cross
            nn.GELU(),  # To use SwiGLU instead, replace with: SwiGLU(hidden_dim, hidden_dim, device=device)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim, device=device),
        )
        
        # Actor projection for latent cross (projects actor_embeds to match history_repr dimension)
        self.actor_projection_for_cross = nn.Linear(embedding_dim, embedding_dim, device=device)
        
        # Target prediction network
        self.get_actor_action_repr = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim, device=device),  # actor_repr + action_emb + interaction
            nn.GELU(),  # To use SwiGLU instead, replace with: SwiGLU(hidden_dim, hidden_dim, device=device)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim, device=device),  # Output D_emb for dot product operations
        )
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.action_embeddings.weight)
    
    def _encode_history_with_transformer(
        self,
        history_actions: torch.Tensor,
        history_targets: torch.Tensor,
        history_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Shared method to encode history using transformer encoder.
        
        This method handles the common logic of:
        1. Getting embeddings for actions and targets
        2. Concatenating them
        3. Applying transformer encoding with masking
        4. Projecting to D_emb dimension
        
        Args:
            history_actions: [B, N] - action IDs (padded with zeros for short histories)
            history_targets: [B, N] - target user IDs (padded with zeros for short histories)
            history_mask: [B, N] - validity mask where 1=valid, 0=padding
                Example: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1] for a user with 2 interactions
                in a sequence of length 16 (valid entries at the end)
            
        Returns:
            torch.Tensor: [B, N, D_emb] - full sequence encoded history
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
            # If all tokens are masked, pass a zero tensor through the projection layer
            # to ensure consistent behavior (e.g., applying bias) and avoid NaNs
            # from the transformer on fully masked inputs.
            zeros_input = torch.zeros_like(history_embeds)
            return self.history_projection(zeros_input)
        
        # Apply transformer encoder
        encoded_history = self.history_encoder(
            history_embeds,
            src_key_padding_mask=attention_mask
        )  # [B, N, D_emb * 2]
        
        # Project to D_emb dimension
        encoded_history = self.history_projection(encoded_history)  # [B, N, D_emb]
        
        return encoded_history

    def _encode_history_with_simple_attention(
        self,
        history_actions: torch.Tensor,
        history_targets: torch.Tensor,
        history_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode history using simple pooled multi-head attention with fixed learnable queries.

        This method uses `self.num_attention_heads` learnable query vectors to compute attention
        over the history sequence. For each position `i`, it attends to positions `0` to `i`
        (causal attention) and produces a weighted representation, using the same set of
        learnable queries for all output positions.

        Args:
            history_actions: [B, N] - action IDs (padded with zeros for short histories)
            history_targets: [B, N] - target user IDs (padded with zeros for short histories)
            history_mask: [B, N] - validity mask where 1=valid, 0=padding

        Returns:
            torch.Tensor: [B, N, D_emb] - full sequence encoded history
        """
        # Get embeddings for each history item
        # action_embeds: [B, N, D_emb]
        action_embeds = self.action_embeddings(history_actions)
        # target_embeds: [B, N, D_emb]
        target_embeds = self.user_embeddings(history_targets)

        # Concatenate action and target embeddings to form the full history embeddings
        # history_embeds: [B, N, D_emb * 2]
        history_embeds = torch.cat([action_embeds, target_embeds], dim=-1)

        batch_size, seq_len, embed_dim_combined = history_embeds.shape
        # embed_dim_combined is D_emb * 2

        # Create a boolean mask for padded tokens: True where token is padding (0), False where valid (1)
        # attention_mask_padding: [B, N]
        attention_mask_padding = (history_mask == 0).bool()

        # Handle the edge case where all tokens in all sequences are masked (e.g., empty batches)
        if attention_mask_padding.all():
            # If all tokens are masked, return a zero tensor of the expected output shape
            # The output shape is [B, N, D_emb]
            return torch.zeros(batch_size, seq_len, self.D_emb, device=self.device)

        # --- Attention Calculation ---

        # 1. Compute raw attention scores between learnable queries and history embeddings (Keys)
        # self.learnable_queries: [num_attention_heads, embed_dim_combined] (K, D_combined)
        # history_embeds: [B, N, embed_dim_combined] (B, N, D_combined)
        # torch.einsum('kd,bnd->kbn', ...) performs (K, D_combined) @ (B, D_combined, N) -> (K, B, N)
        # raw_attention_scores[k, b, j] = dot_product(learnable_queries[k], history_embeds[b, j])
        raw_attention_scores = torch.einsum('kd,bnd->kbn', self.learnable_queries, history_embeds)
        # Scale by sqrt(d_k), where d_k is the dimension of the keys (embed_dim_combined)
        raw_attention_scores = raw_attention_scores / (embed_dim_combined ** 0.5)
        # raw_attention_scores: [num_attention_heads, B, N]

        # 2. Expand raw scores to match the desired attention matrix shape [K, B, N (query_pos), N (key_pos)]
        # For each output position 'i' (the query_pos dimension), we use the same fixed learnable queries.
        # So, we simply replicate the raw scores across the 'query_pos' dimension.
        # attention_scores[k, b, i, j] will be raw_attention_scores[k, b, j]
        attention_scores = raw_attention_scores.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # attention_scores: [num_attention_heads, B, N, N]

        # 3. Create and apply the causal mask
        # causal_mask: [N, N] - True for upper triangle (positions j > i), False otherwise
        # This ensures that for output position 'i', attention is only paid to input positions 'j' <= 'i'.
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
        # Expand causal_mask to [1, 1, N, N] for broadcasting
        attention_scores = attention_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        # attention_scores: [num_attention_heads, B, N, N]

        # 4. Apply the padding mask
        # attention_mask_padding: [B, N] - True where input token is padding
        # We need to mask the 'key_pos' dimension (last N) for padded inputs.
        # Unsqueeze attention_mask_padding to [1, B, 1, N] to broadcast correctly.
        attention_scores = attention_scores.masked_fill(attention_mask_padding.unsqueeze(0).unsqueeze(2), float('-inf'))
        # attention_scores: [num_attention_heads, B, N, N]

        # 5. Apply softmax to get attention weights
        # Softmax is applied along the last dimension (key_pos), ensuring weights sum to 1 for each query-output pair.
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # attention_weights: [num_attention_heads, B, N, N]

        # Replace NaN values with zeros (for numerical stability, especially after softmax on -inf values)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0, posinf=0.0, neginf=0.0)

        # 6. Apply attention weights to get weighted representations (Context Vectors)
        # history_embeds: [B, N, embed_dim_combined] (V)
        # Unsqueeze history_embeds to [1, B, N, embed_dim_combined] for broadcasting with attention_weights
        # weighted_sum = (K, B, N, N) @ (1, B, N, D_combined) -> (K, B, N, D_combined)
        weighted_sum = torch.matmul(attention_weights, history_embeds.unsqueeze(0))
        # weighted_sum: [num_attention_heads, B, N, embed_dim_combined]

        # 7. Reshape and concatenate all attention head outputs
        # Transpose to [B, num_attention_heads, N, embed_dim_combined]
        weighted_sum = weighted_sum.transpose(0, 1)
        # Reshape to [B, N, num_attention_heads * embed_dim_combined]
        # This concatenates the output from each learnable query (head) for each (B, N) position.
        weighted_sum = weighted_sum.reshape(batch_size, seq_len, self.num_attention_heads * embed_dim_combined)

        # 8. Project to final embedding dimension D_emb
        # encoded_history: [B, N, D_emb]
        encoded_history = self.simple_attention_projection(weighted_sum)

        return encoded_history

    def encode_history_for_target(
        self,
        history_actions: torch.Tensor,
        history_targets: torch.Tensor,
        history_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode user's interaction history for target prediction using transformer encoder.
        
        This method handles variable-length histories using a mask. The transformer
        will only attend to valid (masked) positions, and the final representation
        is computed as a mean over valid tokens only.
        
        Data Flow:
        1. Each history item: [action_id, target_user_id] 
        2. Embeddings: [action_embedding(D_emb), target_user_embedding(D_emb)]
        3. Concatenated: [action_features + target_features] = [D_emb * 2]
        4. Transformer processes sequence of these concatenated embeddings
        5. Projection to D_emb dimension: [B, N, D_emb]
        6. Mean pooling over valid tokens: [B, D_emb]
        
        Args:
            history_actions: [B, N] - action IDs (padded with zeros for short histories)
            history_targets: [B, N] - target user IDs (padded with zeros for short histories)
            history_mask: [B, N] - validity mask where 1=valid, 0=padding
                Example: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1] for a user with 2 interactions
                in a sequence of length 16 (valid entries at the end)
            
        Returns:
            torch.Tensor: [B, D_emb] - encoded history representation for target prediction
        """
        # Get full sequence from appropriate encoding method
        if self.history_encoder_type == "simple_attention":
            encoded_history = self._encode_history_with_simple_attention(history_actions, history_targets, history_mask)  # [B, N, D_emb]
        else:
            encoded_history = self._encode_history_with_transformer(history_actions, history_targets, history_mask)  # [B, N, D_emb]
        
        # Mean pooling over valid tokens
        masked_encoded = encoded_history * history_mask.unsqueeze(-1)  # [B, N, D_emb]
        valid_tokens = history_mask.sum(dim=1, keepdim=True)  # [B, 1]
        valid_tokens = torch.clamp(valid_tokens, min=1)  # Avoid division by zero
        
        history_repr = masked_encoded.sum(dim=1) / valid_tokens  # [B, D_emb]
        
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
        history_repr = self.encode_history_for_target(
            actor_history_actions, 
            actor_history_targets, 
            actor_history_mask
        )  # [B, D_emb]

        # Combine actor and history representations with latent cross interaction
        # Inspired by Google's "latent cross" approach: concatenate + elementwise product
        actor_history_concat = torch.cat([actor_embeds, history_repr], dim=-1)  # [B, D_emb + D_emb]
        
        # Project actor embeddings to match history representation dimension for latent cross
        actor_embeds_projected = self.actor_projection_for_cross(actor_embeds)  # [B, D_emb]
        actor_history_cross = actor_embeds_projected * history_repr  # [B, D_emb] - elementwise product
        
        # Concatenate all three: actor_embeds, history_repr, and their elementwise product
        actor_history_input = torch.cat([actor_history_concat, actor_history_cross], dim=-1)  # [B, D_emb + D_emb + D_emb]
        actor_repr = self.actor_projection(actor_history_input)  # [B, D_emb]
        
        # Get action embeddings
        action_embeds = self.action_embeddings(example_action)  # [B, D_emb]
        
        # Create actor-action representation: concatenate actor_repr, action_repr, and their elementwise product
        actor_action_repr = torch.cat([
            actor_repr, 
            action_embeds, 
            actor_repr * action_embeds  # Elementwise product for interaction modeling
        ], dim=-1)  # [B, D_emb * 3]
        
        # Pass through MLP to get final actor-action representation
        actor_action_repr = self.get_actor_action_repr(actor_action_repr)  # [B, D_emb]
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
        
        batch_size = batch.actor_id.shape[0]
        if num_rand_negs > 0:
            # Add num_rand_negs additional random negative targets
            random_neg_targets = torch.randint(
                0, self.num_users,
                (batch_size, num_rand_negs),
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
            random_neg_mask = torch.ones(batch_size, num_rand_negs, device=self.device)  # [B, num_rand_negs]
            for i in range(batch_size):
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
        labels = torch.arange(batch_size, device=self.device)  # [B]
        
        # Calculate loss using cross-entropy
        loss = torch.nn.functional.cross_entropy(masked_logits, labels)
        
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
        Efficient temporal pretraining loss using causal attention.
        
        This method creates multiple training examples from a single interaction sequence:
        - Uses causal attention to ensure each position only sees previous positions
        - Extracts representations at each position efficiently (no re-encoding)
        - Batches all temporal predictions together for efficiency
        
        This provides much more training signal and helps learn temporal patterns.
        
        Args:
            batch: NextTargetPredictionBatch containing training data
            num_temporal_examples: Number of temporal examples to create per batch item
            
        Returns:
            Dict containing temporal loss and metrics
        """
        batch_size = batch.actor_history_actions.size(0)
        history_length = batch.actor_history_actions.size(1)

        # Use appropriate method to get full sequence for temporal pretraining
        if self.history_encoder_type == "simple_attention":
            encoded_history = self._encode_history_with_simple_attention(
                batch.actor_history_actions,
                batch.actor_history_targets,
                batch.actor_history_mask
            )  # [B, N, D_emb]
        else:
            encoded_history = self._encode_history_with_transformer(
                batch.actor_history_actions,
                batch.actor_history_targets,
                batch.actor_history_mask
            )  # [B, N, D_emb]

        # Ensure we have enough history for temporal examples
        if history_length < num_temporal_examples + 1:
            num_temporal_examples = max(1, history_length - 1)

        # Determine which positions to use for temporal prediction
        # We want the last num_temporal_examples positions that have valid next actions
        temporal_positions = []
        for i in range(num_temporal_examples):
            pos = history_length - num_temporal_examples + i
            if pos >= 0 and pos < history_length - 1:  # Need at least one position after for prediction
                temporal_positions.append(pos)
        
        if not temporal_positions:
            # No valid temporal positions
            return {
                'temporal_loss': torch.tensor(0.0, device=self.device),
                'temporal_accuracy': torch.tensor(0.0, device=self.device),
                'num_temporal_examples': 0,
            }
        
        # Extract representations at temporal positions
        temporal_reprs = encoded_history[:, temporal_positions]  # [B, num_temporal, D_emb]
        temporal_actions = batch.actor_history_actions[:, [p+1 for p in temporal_positions]]  # [B, num_temporal]
        temporal_targets = batch.actor_history_targets[:, [p+1 for p in temporal_positions]]  # [B, num_temporal]
        temporal_valid = batch.actor_history_mask[:, [p+1 for p in temporal_positions]]  # [B, num_temporal]
        
        # Additional validation: ensure temporal targets are within valid range
        temporal_valid = temporal_valid.bool() & (temporal_targets < self.num_users) & (temporal_targets >= 0)
        
        # Get actor embeddings (same for all temporal positions)
        actor_embeds = self.user_embeddings(batch.actor_id)  # [B, D_emb]
        actor_embeds_expanded = actor_embeds.unsqueeze(1).expand(-1, len(temporal_positions), -1)  # [B, num_temporal, D_emb]
        
        # Combine actor and temporal history representations with latent cross interaction
        # Inspired by Google's "latent cross" approach: concatenate + elementwise product
        actor_temporal_concat = torch.cat([actor_embeds_expanded, temporal_reprs], dim=-1)  # [B, num_temporal, D_emb + D_emb]
        
        # Project actor embeddings to match temporal representation dimension for latent cross
        actor_embeds_projected = self.actor_projection_for_cross(actor_embeds)  # [B, D_emb]
        actor_embeds_projected_expanded = actor_embeds_projected.unsqueeze(1).expand(-1, len(temporal_positions), -1)  # [B, num_temporal, D_emb]
        actor_temporal_cross = actor_embeds_projected_expanded * temporal_reprs  # [B, num_temporal, D_emb] - elementwise product
        
        # Concatenate all three: actor_embeds, temporal_reprs, and their elementwise product
        actor_temporal_input = torch.cat([actor_temporal_concat, actor_temporal_cross], dim=-1)  # [B, num_temporal, D_emb + D_emb + D_emb]
        actor_reprs = self.actor_projection(actor_temporal_input)  # [B, num_temporal, D_emb]
        
        # Get action embeddings for temporal actions
        action_embeds = self.action_embeddings(temporal_actions)  # [B, num_temporal, D_emb]
        
        # Create actor-action representations
        actor_action_input = torch.cat([
            actor_reprs, 
            action_embeds, 
            actor_reprs * action_embeds  # Elementwise product
        ], dim=-1)  # [B, num_temporal, D_emb * 3]
        
        # Final actor-action representations
        actor_action_reprs = self.get_actor_action_repr(actor_action_input)  # [B, num_temporal, D_emb]
        
        # Get target embeddings for all users
        all_target_embeds = self.user_embeddings.weight  # [num_users, D_emb]
        
        # Compute logits for all temporal positions at once
        # Reshape to [B * num_temporal, D_emb] for efficient matrix multiplication
        actor_action_reprs_flat = actor_action_reprs.view(-1, self.embedding_dim)  # [B * num_temporal, D_emb]
        logits = torch.matmul(actor_action_reprs_flat, all_target_embeds.t())  # [B * num_temporal, num_users]
        
        # Reshape back to [B, num_temporal, num_users]
        logits = logits.view(batch_size, len(temporal_positions), self.num_users)  # [B, num_temporal, num_users]
        
        # Note: We do NOT mask the logits here because cross_entropy automatically
        # handles the positive class selection via the target argument.
        # Masking would prevent the model from learning the correct positive class.
        masked_logits = logits  # [B, num_temporal, num_users]
        
        # Flatten for loss computation
        masked_logits_flat = masked_logits.view(-1, self.num_users)  # [B * num_temporal, num_users]
        temporal_targets_flat = temporal_targets.view(-1)  # [B * num_temporal]
        temporal_valid_flat = temporal_valid.view(-1)  # [B * num_temporal]
        
        # Calculate loss only for valid examples
        valid_indices = torch.where(temporal_valid_flat)[0]
        if len(valid_indices) > 0:
            valid_logits = masked_logits_flat[valid_indices]  # [valid_count, num_users]
            valid_targets = temporal_targets_flat[valid_indices]  # [valid_count]
            
            # Cross-entropy loss
            loss = torch.nn.functional.cross_entropy(valid_logits, valid_targets)
            
            # Accuracy
            predictions = torch.argmax(valid_logits, dim=1)
            accuracy = (predictions == valid_targets).float().mean()
            
            num_temporal_examples_used = len(valid_indices)
        else:
            loss = torch.tensor(0.0, device=self.device)
            accuracy = torch.tensor(0.0, device=self.device)
            num_temporal_examples_used = 0
        
        return {
            'temporal_loss': loss,
            'temporal_accuracy': accuracy,
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
        combined_loss = ((1 - temporal_weight) * standard_results['loss'] +
                         temporal_weight * temporal_results['temporal_loss'])

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