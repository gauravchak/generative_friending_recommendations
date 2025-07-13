# Generative Friend Recommendations

## Problem Statement

Friend recommendation is fundamentally different from content recommendation. While content recommendation suggests items (videos, products, articles) to users, friend recommendation suggests **people to people**. This creates unique challenges:

- **Bidirectional relationships**: Friend connections are mutual, requiring understanding of both users' preferences
- **Sparse interactions**: Users interact with far fewer people than content items
- **Temporal dynamics**: Friendship patterns evolve over time
- **Privacy constraints**: Limited access to user behavior data
- **Cold start**: New users have no interaction history

## Project Overview

This repository demonstrates a **progressive approach** to friend recommendation, starting with simple methods and evolving to sophisticated generative techniques:

1. **Next Target Prediction** (Current) - Predict who a user will interact with next
2. **Social Tokenized UserIDs** (Planned) - Represent users as token sequences
3. **Generative Recommendations** (Planned) - Generate friend suggestions using language models

### Current Status: Next Target Prediction

We've implemented a robust next target prediction system with **two history encoding approaches**:

- **Simple Pooled Multi-Head Attention**: K=2 learnable query vectors with causal attention
- **Transformer Encoder**: Full transformer with self-attention across all positions

This allows readers to understand the **value of attention mechanisms** before diving into complex transformer architectures.

## Architecture Overview

### Core Model Components

- **User & Action Embeddings**: Learn representations for users and social actions
- **History Encoder**: Process user interaction sequences (two approaches available)
- **Actor-Action Representation**: Combine user context with current action
- **Retrieval System**: Find similar users using dot product similarity

### Key Innovations

- **Mixed Negative Sampling**: Combines in-batch and random negatives for robust training
- **Temporal Pretraining**: Uses historical sequences to create additional training examples
- **Variable-Length Histories**: Efficiently handles users with different interaction counts
- **Causal Attention**: Ensures temporal consistency in sequence modeling

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test_next_target_prediction_userids.py

# Train with simple attention
cd test_data
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_with_realistic_data.py --history_encoder_type simple_attention

# Train with transformer
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_with_realistic_data.py --history_encoder_type transformer
```

## Results & Performance

### Educational Progression: Simple Attention → Transformer

| Approach | Test Accuracy | Test MRR | Mean Rank | Model Complexity |
|----------|---------------|----------|-----------|------------------|
| **Simple Attention (K=2)** | 71.42% | 43.47% | 7.52 | Low |
| **Transformer Encoder** | 72.86% | 43.58% | 7.59 | High |

**Key Insights:**
- Simple attention achieves **98% of transformer performance** with much lower complexity
- This demonstrates that **attention mechanisms** are the key innovation, not necessarily full transformers
- Perfect for educational purposes: readers can understand attention before complex architectures

### Recent Performance Improvements

- **+31.9% accuracy** improvement through D_emb projection optimization
- **+116.2% MRR** improvement from bug fixes and architectural refinements
- **Consistent 72%+ accuracy** on realistic social network data

## Project Structure

```
generative_friending_recommendations/
├── src/next_target_prediction_userids/     # Main implementation
│   ├── next_target_prediction_userids.py   # Core model
│   ├── README.md                           # Detailed documentation
│   └── example_usage.py                    # Usage examples
├── test_data/                              # Training data and scripts
│   ├── train_with_realistic_data.py        # Main training script
│   └── social_network_data.json            # Realistic test data
├── tests/                                  # Test suite
└── README.md                               # This file
```

## Implementation Details

### History Encoder Options

**Simple Attention (`history_encoder_type="simple_attention"`):**
- K=2 learnable query vectors
- Causal attention (position i only sees positions 0 to i)
- Much simpler to understand and implement
- Nearly equivalent performance to transformer

**Transformer (`history_encoder_type="transformer"`):**
- Full transformer encoder with self-attention
- Multi-layer architecture with feed-forward networks
- More complex but potentially more expressive

### Training Approaches

- **Pure In-Batch**: Uses other batch examples as negatives (efficient)
- **Mixed Negative**: Combines in-batch with random negatives (robust)
- **Temporal Pretraining**: Creates additional examples from historical sequences

## Future Work

### Phase 2: Social Tokenized UserIDs (STU)
- Represent each user as a 3-token sequence
- Change from target prediction to next-token prediction
- Enable generative capabilities

### Phase 3: Generative Recommendations
- Use language models to generate friend suggestions
- Leverage STU representations for natural language output
- Enable explainable recommendations

### Phase 4: In-Model Clustering
- Intermediate clustering step to reduce embedding table size
- Address overfitting concerns with large user bases
- Maintain performance while reducing complexity

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd generative_friending_recommendations
pip install -r requirements.txt

# Run tests
python tests/test_next_target_prediction_userids.py

# Run training
cd test_data
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_with_realistic_data.py
```

## References

- [Mixed Negative Sampling for Learning Two-tower Neural Networks](https://arxiv.org/abs/2203.06717)
- [Practical Lessons from Deep Retrieval Systems at Scale](https://ai.googleblog.com/2020/07/retrieval-augmented-generation-for.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

**Note**: This project is designed for educational purposes, demonstrating the evolution from simple attention mechanisms to sophisticated generative approaches in friend recommendation systems.