# Social Tokenized UserIDs (STU) Model

## Overview

The STU model represents Phase 2 of our friend recommendation progression, moving from traditional two-tower retrieval to sequence prediction. Instead of learning user embeddings and computing dot product similarity, we now represent users as 3-token sequences and predict target users autoregressively.

## Key Concepts

### **Social Tokenized UserIDs**
- Each user is represented by exactly `num_codebooks` integers: `[token_0, token_1, ..., token_{num_codebooks-1}]`
- These tokens are derived from hierarchical clustering (assumed to be pre-computed via RQVAE)
- Vocabulary size: 10,000 tokens per level
- Total possible users: 10,000^num_codebooks (configurable complexity)
- Default: 3 tokens (10,000³ = 1 trillion users)

### **Autoregressive Prediction**
Instead of predicting a single target, we predict `num_codebooks` tokens sequentially:
1. **Tower 0**: Predict `token_0` given user context
2. **Tower 1**: Predict `token_1` given user context + `token_0`
3. **Tower 2**: Predict `token_2` given user context + `token_0` + `token_1`
4. **...**: Continue for all `num_codebooks` tokens

### **Teacher Forcing**
During training, we use ground truth tokens for conditioning:
- Tower 0: `P(token_0 | user_context)`
- Tower 1: `P(token_1 | user_context, target_token_0)`
- Tower 2: `P(token_2 | user_context, target_token_0, target_token_1)`
- Tower i: `P(token_i | user_context, target_token_0, ..., target_token_{i-1})`

## Architecture

### **Shared Components**
```python
def get_user_action_repr(self, actor_stu, history_actions, history_targets, history_mask, example_action):
    # 1. Encode actor STU (mean pooling over num_codebooks tokens)
    # 2. Encode history (masked mean pooling over actions and target STUs)
    # 3. Encode example action
    # 4. Combine and pass through MLP
    return user_action_repr [B, hidden_dim]
```

### **Unified Tower Components**
```python
def predict_token(self, user_action_repr, previous_tokens, token_idx):
    # Encode previous tokens + concatenate + tower_layers[token_idx] → token_probs [B, vocab_size]

# Tower layers are created dynamically:
# tower_layers[0]: hidden_dim → vocab_size
# tower_layers[1]: hidden_dim + embedding_dim → vocab_size  
# tower_layers[2]: hidden_dim + 2*embedding_dim → vocab_size
# tower_layers[i]: hidden_dim + i*embedding_dim → vocab_size
```

## Data Structure

### **STUBatch**
```python
@dataclass
class STUBatch:
    actor_stu: torch.Tensor              # [B, num_codebooks] - Current user's STU tokens
    actor_history_actions: torch.Tensor  # [B, N] - Historical action IDs
    actor_history_targets: torch.Tensor  # [B, num_codebooks*N] - Historical target STUs
    actor_history_mask: torch.Tensor     # [B, N] - Validity mask for variable-length histories
    example_action: torch.Tensor         # [B] - Current action ID
    example_target_stu: torch.Tensor     # [B, num_codebooks] - Target user's STU tokens
```

### **Key Differences from UserID Model**
- **Input**: STU tokens instead of user IDs
- **History**: Target STUs are `num_codebooks*N` instead of N (`num_codebooks` tokens per target)
- **Output**: `num_codebooks` separate token predictions instead of similarity score
- **Loss**: `num_codebooks` cross-entropy losses instead of contrastive loss

## Training

### **Loss Function**
```python
total_loss = loss_0 + loss_1 + ... + loss_{num_codebooks-1}
```
Where each loss is cross-entropy between predicted and target tokens.

### **Metrics**
- **Individual accuracies**: `accuracy_0`, `accuracy_1`, ..., `accuracy_{num_codebooks-1}`
- **Overall accuracy**: All `num_codebooks` tokens correct
- **Individual losses**: `loss_0`, `loss_1`, ..., `loss_{num_codebooks-1}`

## Simplifications in This Implementation

### **Mean Pooling Instead of Attention**
- Actor STU: Mean pooling over `num_codebooks` tokens
- History: Mean pooling over all historical tokens
- Simple but effective baseline

### **No Temporal Pretraining**
- Focus on basic autoregressive prediction
- Can be added later for comparison

### **No Inference Methods**
- Training only for now
- Beam search and other inference methods to be added later

## Parametrization Benefits

### **Configurable Complexity**
- **Flexible token count**: Use 2, 3, 4, or more tokens per user
- **Adaptive vocabulary**: Scale from 10,000² (100M users) to 10,000⁴ (10¹⁶ users)
- **Memory efficiency**: Choose appropriate complexity for your user base

### **Unified Architecture**
- **Single `predict_token` method**: Eliminates code duplication across towers
- **Dynamic tower creation**: `tower_layers` created based on `num_codebooks`
- **Consistent interface**: Same API regardless of token count

### **Easy Experimentation**
- **A/B test token counts**: Compare 2 vs 3 vs 4 tokens
- **Progressive complexity**: Start simple, scale up as needed
- **Backward compatibility**: Default 3 tokens maintains existing behavior

## Future Enhancements

1. **Attention Mechanisms**: Replace mean pooling with transformer attention
2. **Temporal Pretraining**: Add history-based pretraining objectives
3. **Beam Search**: Implement autoregressive inference
4. **MoE Towers**: Add mixture of experts to each tower
5. **Advanced Negative Sampling**: Improve training with better negatives

## Usage

```python
from next_target_prediction_stu import NextTargetPredictionSTU, STUBatch

# Initialize model (default: 3 codebooks)
model = NextTargetPredictionSTU(
    num_actions=5,
    num_codebooks=3,  # Configurable number of tokens per user
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=256,
    device="cpu"
)

# Or with different number of codebooks
model_4_tokens = NextTargetPredictionSTU(
    num_actions=5,
    num_codebooks=4,  # 4 tokens per user
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=256,
    device="cpu"
)

# Create batch
batch = STUBatch(...)

# Training forward pass
results = model.train_forward_with_target(batch)
print(f"Loss: {results['loss']}")
print(f"Overall accuracy: {results['overall_accuracy']}")
```

## Progression to Phase 3

This STU model sets up the foundation for generative recommendations:
- **Token-based representation**: Users as sequences instead of embeddings
- **Autoregressive prediction**: Sequential generation like language models
- **Vocabulary-based output**: Fixed vocabulary instead of continuous space

The next phase will use these token sequences to generate natural language friend recommendations. 