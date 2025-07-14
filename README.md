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

# Train on regular dataset with MLP (simpler, faster)
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_data/train_friend_recommendation.py --dataset regular --history_encoder_type transformer --interaction_type mlp

# Train on regular dataset with MoE (best performance)
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_data/train_friend_recommendation.py --dataset regular --history_encoder_type transformer --interaction_type moe --num_experts 4

# Train on 4x larger dataset with MoE
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_data/train_friend_recommendation.py --dataset 4x --interaction_type moe --num_experts 4

# Train with simple attention (educational)
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_data/train_friend_recommendation.py --dataset regular --history_encoder_type simple_attention
```

## Unified Training Script

The `test_data/train_friend_recommendation.py` script provides a unified interface for training on both datasets:

### Key Features
- **Dataset Selection**: Choose between `regular` and `4x` datasets
- **Auto-detection**: Automatically detects optimal hyperparameters for each dataset
- **Flexible Configuration**: Override any hyperparameter via command line arguments
- **Consistent Interface**: Same training pipeline for both datasets
- **Organized Structure**: Datasets are stored in `test_data/regular/` and `test_data/4x/`

### Dataset-Specific Optimizations
- **Regular Dataset**: 32 batch size, 128 embedding dim, 256 hidden dim, Adam optimizer
- **4x Dataset**: 128 batch size, 256 embedding dim, 512 hidden dim, AdamW optimizer

### Command Line Options
```bash
--dataset {regular,4x}           # Dataset to use
--data_dir PATH                  # Custom data directory
--history_encoder_type {transformer,simple_attention}
--interaction_type {mlp,moe}     # Interaction modeling approach
--num_experts INT                # Number of MoE experts
--num_epochs INT                 # Training epochs
--batch_size INT                 # Batch size
--learning_rate FLOAT            # Learning rate
--embedding_dim INT              # Embedding dimension
--hidden_dim INT                 # Hidden dimension
--output_dir PATH                # Output directory
```

## Results & Performance

### Simplified Architecture: MLP vs MoE Interaction Modeling

| Approach | Test Accuracy | Test MRR | Mean Rank | Parameters | Training Time/Epoch |
|----------|---------------|----------|-----------|------------|-------------------|
| **MLP (Simplified)** | 77.33% | 46.21% | 11.29 | 1,278,080 | ~24s |
| **MoE (4 Experts)** | **81.91%** | **59.22%** | **8.40** | 1,950,980 | ~25s |

**Key Insights:**
- **MoE clearly outperforms MLP** with +4.58% accuracy and +13.01% MRR
- **Excellent cost-performance trade-off**: 53% more parameters for 28% better MRR
- **Minimal training overhead**: Only 4% slower training for significant performance gains
- **Production-ready results**: 81.91% accuracy and 59.22% MRR are excellent for friend recommendation

### MLP Performance Improvements vs Previous Architecture

**New MLP Results**: 77.33% accuracy, 46.21% MRR, 11.29 mean rank
- **+5.53% accuracy** improvement vs previous transformer (71.80% → 77.33%)
- **+13.07% MRR** improvement vs previous transformer (33.14% → 46.21%)
- **Better mean rank**: -2.82 improvement (14.11 → 11.29)
- **Efficiency gains**: -6% parameters, -4% training time

### MoE Performance Highlights

**New MoE Results**: 81.91% accuracy, 59.22% MRR, 8.40 mean rank
- **+4.58% accuracy** vs simplified MLP (77.33% → 81.91%)
- **+13.01% MRR** vs simplified MLP (46.21% → 59.22%)
- **Better ranking**: -2.89 mean rank improvement (11.29 → 8.40)
- **Specialized learning**: 4 experts learn different interaction patterns
- **Smart routing**: Gating network learns to route inputs to appropriate experts

### Architectural Innovation

The simplified architecture with unified D_emb*5 representation enables both MLP and MoE options:
- **Unified representation**: [actor, history, action, actor*history, actor*action]
- **Flexible interaction modeling**: Choose between simple MLP or sophisticated MoE
- **Consistent architecture**: Same representation for main training and temporal pretraining
- **Better parameter efficiency**: Eliminated redundant projection layers

## Project Structure

```
generative_friending_recommendations/
├── src/next_target_prediction_userids/     # Main implementation
│   ├── next_target_prediction_userids.py   # Core model
│   ├── README.md                           # Detailed documentation
│   └── example_usage.py                    # Usage examples
├── test_data/                              # Training data and scripts
│   ├── train_friend_recommendation.py      # Unified training script
│   ├── regular/                            # Regular dataset
│   │   ├── social_network_data.json        # Regular test data
│   │   ├── data_loader.py                  # Data loader for regular dataset
│   │   ├── best_model.pth                  # Trained model
│   │   └── training_history.json           # Training history
│   └── 4x/                                 # 4x larger dataset
│       ├── social_network_data_4x.json     # Larger test data
│       ├── best_model_4x.pth               # Trained model
│       └── training_history_4x.json        # Training history
├── tests/                                  # Test suite
└── README.md                               # This file
```

## Implementation Details

### History Encoder Options

**Simple Attention (`history_encoder_type="simple_attention"`):**
- K=2 learnable query vectors
- Causal attention (position i only sees positions 0 to i)
- Much simpler to understand and implement
- Perfect for educational purposes

**Transformer (`history_encoder_type="transformer"`):**
- Full transformer encoder with self-attention
- Multi-layer architecture with feed-forward networks
- Best performance when combined with MoE interaction modeling

### Interaction Modeling Options

**MLP (`interaction_type="mlp"`):**
- Simple 2-layer neural network
- Efficient and easy to understand
- Good performance with reasonable computational cost
- Recommended for development and resource-constrained environments

**Mixture of Experts (`interaction_type="moe"`):**
- Multiple expert networks with learned gating
- Specialized experts for different interaction patterns
- Superior performance with moderate computational overhead
- Recommended for production use
- Configurable number of experts (`num_experts` parameter)

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
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_data/train_friend_recommendation.py --dataset regular
```

## References

- [Mixed Negative Sampling for Learning Two-tower Neural Networks](https://arxiv.org/abs/2203.06717)
- [Practical Lessons from Deep Retrieval Systems at Scale](https://ai.googleblog.com/2020/07/retrieval-augmented-generation-for.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 4X Dataset Results

### Realistic Social Network Dataset (8K users, 9K actions)

The 4X dataset provides a more realistic challenge with larger scale and more complex social patterns:

| Approach | Test Accuracy | Test MRR | Parameters | Training Time/Epoch | Dataset |
|----------|---------------|----------|------------|-------------------|---------|
| **MLP (4X Dataset)** | 43.47% | 29.45% | 6,123,776 | ~55s | 8K users, 9K actions |
| **MoE (4X Dataset)** | TBD | TBD | ~8.8M | TBD | 8K users, 9K actions |

**Key Insights:**
- **Much more realistic challenge**: 43.47% accuracy is excellent for real-world friend recommendation
- **Larger model required**: 6.1M parameters vs 1.3M for original dataset
- **Realistic performance**: 29.45% MRR reflects the difficulty of predicting real social patterns
- **Overfitting challenges**: Large gap between training and test performance indicates need for regularization

### Dataset Characteristics

**4X Dataset Features:**
- **8,000 users** (4x larger than original)
- **9,148 actions** (more realistic distribution)
- **4,279 training examples** (sparse interactions)
- **Realistic patterns**: Age-based, geographic, and interest-based homophily
- **Temporal dynamics**: Hour-of-day and day-of-week patterns
- **Power law distributions**: Realistic user popularity and activity levels

### Performance Comparison

| Metric | Original Dataset | 4X Dataset | Notes |
|--------|------------------|------------|-------|
| **Users** | 2,000 | 8,000 | 4x larger |
| **Actions** | ~20,000 | 9,148 | More realistic |
| **Training Examples** | ~18,000 | 4,279 | Sparser interactions |
| **Model Parameters** | 1.28M | 6.12M | 5x larger model |
| **Test Accuracy** | 77.33% (MLP) | 43.47% (MLP) | Much harder task |
| **Test MRR** | 46.21% (MLP) | 29.45% (MLP) | Realistic challenge |

### Training the 4X Dataset

```bash
# Train with MLP on 4X dataset
cd test_data_4x
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_with_4x_data.py --interaction_type mlp --num_epochs 20

# Train with MoE on 4X dataset (when available)
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_with_4x_data.py --interaction_type moe --num_experts 4 --num_epochs 20
```

---

**Note**: This project is designed for educational purposes, demonstrating the evolution from simple attention mechanisms to sophisticated generative approaches in friend recommendation systems.