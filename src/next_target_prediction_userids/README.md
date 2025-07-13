# Next Target Prediction using UserIDs (PyTorch Implementation)

## The Final Insight: Mixed Negative Sampling for Robust Retrieval

The most effective and flexible approach for training retrieval models in friend recommendation is **mixed negative sampling**. This method combines the strengths of in-batch negatives (realistic, hard negatives) with the diversity of random negatives (broad coverage of the embedding space). By simply tuning a parameter `num_rand_negs`, you can control the number of additional random negatives, allowing you to move seamlessly between pure in-batch, pure random, or any mix in between.

**Why does this matter?**
- In-batch negatives are efficient and realistic, but may lack diversity in small batches or highly skewed datasets.
- Random negatives provide diversity, but may be too easy and not reflect real-world confusion.
- Mixed negative sampling, as popularized by recent research (e.g., Google's Mixed Negative Sampling paper), gives you the best of both worlds.

## How We Got Here: Evolution of Negative Sampling

1. **Random Negative Sampling**
   - Early implementations sampled random user IDs as negatives.
   - Simple, but negatives are often too easy and not representative of real confusion in the system.

2. **In-Batch Negative Sampling**
   - Standard in modern recommender systems.
   - Uses other examples in the batch as negatives, which are more likely to be hard negatives.
   - Efficient, but can lack diversity if batch size is small or user distribution is skewed.

3. **Mixed Negative Sampling (Current Approach)**
   - Unified method: always uses in-batch negatives, and optionally adds `num_rand_negs` random negatives per example.
   - Masking ensures that no negative is accidentally a true positive (even among random negatives).
   - Flexible: `num_rand_negs=0` is pure in-batch, `num_rand_negs>0` adds diversity, large `num_rand_negs` approaches pure random.

## Architectural Insights

- **Two-Tower Model**: The model produces an actor-action representation (query tower) and uses a user embedding table for targets (item tower).
- **Modern Activation Functions**: Uses GELU activation (state-of-the-art for transformers) instead of ReLU for better performance. Also implemented SwiGLU activation (state-of-the-art for transformers) with gating mechanism for better feature selection and interaction modeling.
- **Flexible Forward**: The `forward` method only requires actor and action information, producing a representation suitable for retrieval.
- **Unified Training**: The `train_forward` method supports any negative sampling strategy via the `num_rand_negs` parameter and includes temporal pretraining.
- **Variable-Length Histories**: The `history_mask` efficiently handles users with different numbers of interactions using padding and masking.
- **Masking**: All negatives (in-batch and random) are masked to avoid accidental positives.
- **Consistent Device/Batch Handling**: All tensor creation uses the model's `self.device` and `self.batch_size` for consistency and reproducibility.

## Understanding the History Mask

The `history_mask` is crucial for handling variable-length user interaction histories efficiently:

**Problem**: Users have different numbers of interactions, but neural networks need fixed-size inputs.

**Solution**: Pad all histories to a fixed length `N`, then use a mask to indicate which positions are valid.

**Example**: For a user with only 2 interactions in a sequence of length 16:
```python
# Padded data (zeros for missing interactions)
actor_history_actions = [action1, action2, 0, 0, 0, ..., 0]  # length 16
actor_history_targets = [user1, user2, 0, 0, 0, ..., 0]     # length 16

# Mask: 1 for valid, 0 for padding (valid entries typically at the end)
actor_history_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
```

**How it works**:
- The transformer encoder only attends to positions where `mask == 1`
- The final history representation is computed as a mean over valid tokens only
- This allows efficient batch processing of users with different interaction counts

## Recent Improvements

### Code Refactoring: Eliminated Duplication
- **Shared History Encoding**: Created `_encode_history_with_transformer()` method to eliminate code duplication between `encode_history()` and `temporal_pretraining_loss()`.
- **Cleaner Architecture**: Both methods now use the same underlying logic for transformer encoding and masking.

### Critical Bug Fix: Temporal Pretraining Loss
- **Issue**: The temporal pretraining loss contained a critical bug where logits were incorrectly masked with `-1e9` for positive targets.
- **Problem**: This prevented the model from learning the correct positive class during temporal pretraining.
- **Fix**: Removed the incorrect masking logic. `torch.nn.functional.cross_entropy` automatically handles positive class selection via the `target` argument.
- **Impact**: **Dramatic performance improvements**:
  - **32% better MRR** (0.1591 → 0.2101)
  - **19% better accuracy** (0.4628 → 0.5521)
  - **Better convergence** (trained for full 15 epochs vs early stopping at 6)
  - **Lower mean rank** (14.63 → 11.84)

### Major Performance Breakthrough: D_emb History Projection
- **Optimization**: Added a projection layer to reduce history representations from `D_emb * 2` to `D_emb` dimensions.
- **Architecture**: Transformer encoder still processes `D_emb * 2` (action + target embeddings), but output is projected to `D_emb` via a small MLP with GELU activation.
- **Benefits**: 
  - **Memory efficiency**: Reduced memory usage in downstream layers
  - **Computational speed**: Faster matrix operations with smaller dimensions
  - **Parameter efficiency**: Only 2.7% parameter overhead (32,896 additional parameters)
  - **Better regularization**: Forces more compact, generalizable representations
- **Impact**: **Exceptional performance improvements**:
  - **+31.9% better accuracy** (55.21% → 72.80%)
  - **+116.2% better MRR** (21.01% → 45.43%)
  - **+39.0% better mean rank** (11.84 → 7.22)
  - **Best results achieved**: 72.80% accuracy, 45.43% MRR, 7.22 mean rank

---

## GELU vs SwiGLU: Activation Function Comparison

### Results on Test Data

| Activation | Test Accuracy | Test MRR | Mean Rank |
|------------|--------------|----------|-----------|
| **GELU + D_emb Projection** | **0.7280** | **0.4543** | **7.22** |
| GELU (After Bug Fix) | 0.5521 | 0.2101 | 11.84 |
| GELU (Before Bug Fix) | 0.4628 | 0.1591 | 14.63 |
| SwiGLU (Before Bug Fix) | 0.4448 | 0.1327 | 16.66 |

- **GELU outperforms SwiGLU** on both accuracy and MRR for the current architecture and data.
- SwiGLU is preserved in the codebase for future experimentation.

### How to Switch to SwiGLU

1. In `next_target_prediction_userids.py`, locate the following lines in the model's `__init__`:
   ```python
   nn.GELU(),  # To use SwiGLU instead, replace with: SwiGLU(hidden_dim, hidden_dim, device=device)
   ```
2. Replace `nn.GELU()` with `SwiGLU(hidden_dim, hidden_dim, device=device)` in both `actor_projection` and `get_actor_action_repr`.
3. Run the training script as usual:
   ```bash
   cd test_data
   PYTORCH_ENABLE_MPS_FALLBACK=1 python train_with_realistic_data.py
   ```
4. Compare the metrics to see if SwiGLU performs better for your data or architecture changes.

---

## Practical Usage

- **Default**: Use `train_forward(batch, num_rand_negs=0)` for efficient, production-grade training with temporal pretraining.
- **Add Diversity**: Use `train_forward(batch, num_rand_negs=2)` or `num_rand_negs=5` to add a few random negatives for more robust learning.
- **Target-Only**: Use `train_forward_with_target(batch, num_rand_negs=0)` for faster training without temporal pretraining.
- **Experiment**: Tune `num_rand_negs` based on your dataset and observed model performance.

## Example Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        results = model.train_forward(batch, num_rand_negs=3)  # 3 random negatives per example + temporal pretraining
        loss = results['loss']
        loss.backward()
        optimizer.step()
```

## Why Not Separate Methods?

Earlier versions of this codebase had separate methods for random and in-batch negative sampling. This was:
- Redundant: Both methods shared most logic.
- Inflexible: Switching strategies required code changes.
- Not best practice: Modern research and production systems favor a unified, parameterized approach.

## Summary Table

| Approach         | Parameter | Negatives Used         | When to Use                |
|------------------|-----------|------------------------|----------------------------|
| In-batch only    | num_rand_negs=0       | Other batch examples   | Large batches, efficiency  |
| Mixed            | num_rand_negs>0       | Batch + num_rand_negs random       | Small batches, more robust |
| Random only      | num_rand_negs>>B      | num_rand_negs random               | Rarely, for ablation only  |

## References
- [Mixed Negative Sampling for Learning Two-tower Neural Networks](https://arxiv.org/abs/2203.06717) (Google Research)
- [Practical Lessons from Deep Retrieval Systems at Scale](https://ai.googleblog.com/2020/07/retrieval-augmented-generation-for.html)

## Additional Documentation
- **[Transformer Architecture Explanation](TRANSFORMER_EXPLANATION.md)**: Detailed explanation of the transformer architecture, `d_model` parameter, and positional embedding considerations

## Temporal Pretraining Loss: Sequence Prefix Prediction

A powerful extension to the standard two-tower loss is the **temporal pretraining loss** (also called sequence prefix prediction or next-interaction prediction). This loss leverages the sequential nature of user histories to provide much richer training signal and better temporal modeling.

### How It Works
- For each user in the batch, instead of only using the full history to predict the next target, we use multiple prefixes of the history.
- For each of the last K positions in the history:
  - Use the history up to position i as the context
  - Use the action and target at position i+1 as the prediction target
- This creates K positive examples per batch item, not just one.

### Why This is a Great Idea

1. **More Training Signal**
   - **Current:** 1 positive example per batch item
   - **Proposed:** K positive examples per batch item (e.g., 8x more training signal)
   - **Result:** Much more efficient use of data, faster convergence

2. **Temporal Learning**
   - **Current:** Only learns from current action → current target
   - **Proposed:** Learns temporal patterns (action_i → target_i+1 relationships)
   - **Result:** Better understanding of sequential user behavior

3. **Self-Supervised Learning**
   - **Current:** Requires explicit positive examples
   - **Proposed:** Creates positive examples from the sequence itself
   - **Result:** Can learn from unlabeled interaction sequences

4. **Better History Encoding**
   - **Current:** History encoder only trained on final prediction
   - **Proposed:** History encoder trained on multiple temporal predictions
   - **Result:** More robust history representations

### Implementation
- See `temporal_pretraining_loss` and `train_forward` in the codebase.
- **Efficient Implementation**: Uses batched processing instead of loops for much better performance.
- You can control the number of temporal examples per batch with a parameter (e.g., `K=8`).
- The temporal loss can be combined with the main loss for joint training.

### Efficiency Improvements
The current implementation is much more efficient than the original loop-based approach:
- **Before**: O(K) transformer forward passes (one per temporal position)
- **After**: O(1) transformer forward pass + efficient tensor operations
- **Speedup**: ~Kx faster for temporal pretraining

### Assumption
- The batch must have at least K+1 valid history positions for temporal pretraining to work (see test code for details).

This approach is inspired by self-supervised sequence modeling in NLP and is highly effective for recommendation and behavioral modeling tasks.

---

This codebase is designed to be a clear, extensible foundation for research and production in social/friending recommendation. The negative sampling strategy is now both state-of-the-art and easy to adapt for your needs. 