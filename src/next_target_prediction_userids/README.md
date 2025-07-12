# Next Target Prediction using UserIDs (PyTorch Implementation)

## The Final Insight: Mixed Negative Sampling for Robust Retrieval

The most effective and flexible approach for training retrieval models in friend recommendation is **mixed negative sampling**. This method combines the strengths of in-batch negatives (realistic, hard negatives) with the diversity of random negatives (broad coverage of the embedding space). By simply tuning a parameter `K`, you can control the number of additional random negatives, allowing you to move seamlessly between pure in-batch, pure random, or any mix in between.

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
   - Unified method: always uses in-batch negatives, and optionally adds `K` random negatives per example.
   - Masking ensures that no negative is accidentally a true positive (even among random negatives).
   - Flexible: `K=0` is pure in-batch, `K>0` adds diversity, large `K` approaches pure random.

## Architectural Insights

- **Two-Tower Model**: The model produces an actor-action representation (query tower) and uses a user embedding table for targets (item tower).
- **Flexible Forward**: The `forward` method only requires actor and action information, producing a representation suitable for retrieval.
- **Unified Training**: The `train_forward` method supports any negative sampling strategy via the `K` parameter.
- **Masking**: All negatives (in-batch and random) are masked to avoid accidental positives.
- **Consistent Device/Batch Handling**: All tensor creation uses the model's `self.device` and `self.batch_size` for consistency and reproducibility.

## Practical Usage

- **Default**: Use `train_forward(batch, K=0)` for efficient, production-grade training.
- **Add Diversity**: Use `train_forward(batch, K=2)` or `K=5` to add a few random negatives for more robust learning.
- **Experiment**: Tune `K` based on your dataset and observed model performance.

## Example Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        results = model.train_forward(batch, K=3)  # 3 random negatives per example
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
| In-batch only    | K=0       | Other batch examples   | Large batches, efficiency  |
| Mixed            | K>0       | Batch + K random       | Small batches, more robust |
| Random only      | K>>B      | K random               | Rarely, for ablation only  |

## References
- [Mixed Negative Sampling for Learning Two-tower Neural Networks](https://arxiv.org/abs/2203.06717) (Google Research)
- [Practical Lessons from Deep Retrieval Systems at Scale](https://ai.googleblog.com/2020/07/retrieval-augmented-generation-for.html)

---

This codebase is designed to be a clear, extensible foundation for research and production in social/friending recommendation. The negative sampling strategy is now both state-of-the-art and easy to adapt for your needs. 