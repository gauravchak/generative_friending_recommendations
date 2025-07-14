# Social Tokenized UserIDs (STU) Model

## Overview

The STU model represents Phase 2 of our friend recommendation progression, moving from traditional two-tower retrieval to sequence prediction. Instead of learning user embeddings and computing dot product similarity, we now represent users as 3-token sequences and predict target users autoregressively.

## Key Concepts

### **Social Tokenized UserIDs**
- Each user is represented by exactly 3 integers: `[token_0, token_1, token_2]`
- These tokens are derived from hierarchical clustering (assumed to be pre-computed via RQVAE)
- Vocabulary size: 10,000 tokens per level
- Total possible users: 10,000³ = 1 trillion (much smaller than user IDs)

### **Autoregressive Prediction**
Instead of predicting a single target, we predict 3 tokens sequentially:
1. **Tower 1**: Predict `token_0` given user context
2. **Tower 2**: Predict `token_1` given user context + `token_0`
3. **Tower 3**: Predict `token_2` given user context + `token_0` + `token_1`

### **Teacher Forcing**
During training, we use ground truth tokens for conditioning:
- Tower 1: `P(token_0 | user_context)`
- Tower 2: `P(token_1 | user_context, target_token_0)`
- Tower 3: `P(token_2 | user_context, target_token_0, target_token_1)`

## Architecture

### **Shared Components**
```python
def get_user_action_repr(self, actor_stu, history_actions, history_targets, example_action):
    # 1. Encode actor STU (mean pooling over 3 tokens)
    # 2. Encode history (mean pooling over actions and target STUs)
    # 3. Encode example action
    # 4. Combine and pass through MLP
    return user_action_repr [B, hidden_dim]
```

### **Tower-Specific Components**
```python
def tower_1(self, user_action_repr):
    # Simple MLP: user_action_repr → token_0_probs [B, 10000]

def tower_2(self, user_action_repr, token_0):
    # Encode token_0 + concatenate + MLP → token_1_probs [B, 10000]

def tower_3(self, user_action_repr, token_0, token_1):
    # Encode token_0, token_1 + concatenate + MLP → token_2_probs [B, 10000]
```

## Data Structure

### **STUBatch**
```python
@dataclass
class STUBatch:
    actor_stu: torch.Tensor              # [B, 3] - Current user's STU tokens
    actor_history_actions: torch.Tensor  # [B, N] - Historical action IDs
    actor_history_targets: torch.Tensor  # [B, 3N] - Historical target STUs
    example_action: torch.Tensor         # [B] - Current action ID
    example_target_stu: torch.Tensor     # [B, 3] - Target user's STU tokens
```

### **Key Differences from UserID Model**
- **Input**: STU tokens instead of user IDs
- **History**: Target STUs are 3N instead of N (3 tokens per target)
- **Output**: 3 separate token predictions instead of similarity score
- **Loss**: 3 cross-entropy losses instead of contrastive loss

## Training

### **Loss Function**
```python
total_loss = loss_0 + loss_1 + loss_2
```
Where each loss is cross-entropy between predicted and target tokens.

### **Metrics**
- **Individual accuracies**: `accuracy_0`, `accuracy_1`, `accuracy_2`
- **Overall accuracy**: All 3 tokens correct
- **Individual losses**: `loss_0`, `loss_1`, `loss_2`

## Simplifications in This Implementation

### **Mean Pooling Instead of Attention**
- Actor STU: Mean pooling over 3 tokens
- History: Mean pooling over all historical tokens
- Simple but effective baseline

### **No Temporal Pretraining**
- Focus on basic autoregressive prediction
- Can be added later for comparison

### **No Inference Methods**
- Training only for now
- Beam search and other inference methods to be added later

## Future Enhancements

1. **Attention Mechanisms**: Replace mean pooling with transformer attention
2. **Temporal Pretraining**: Add history-based pretraining objectives
3. **Beam Search**: Implement autoregressive inference
4. **MoE Towers**: Add mixture of experts to each tower
5. **Advanced Negative Sampling**: Improve training with better negatives

## Usage

```python
from next_target_prediction_stu import NextTargetPredictionSTU, STUBatch

# Initialize model
model = NextTargetPredictionSTU(
    num_actions=5,
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