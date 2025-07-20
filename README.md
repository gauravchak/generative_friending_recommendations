# Generative Friend Recommendations

## The Journey: From Traditional Retrieval to Generative Recommendations

Friend recommendation represents a unique challenge in recommender systems - we're suggesting **people to people**, not content to people. This creates fundamental differences from traditional content recommendation:

- **Bidirectional relationships**: Friend connections are mutual, requiring understanding of both users' preferences
- **Highly personalized interactions**: Unlike content (where videos can be watched by millions), friending is deeply personal - the median user has less than 20 bidirectional friends on social networks
- **Delayed reward**: Unlike content interactions (immediate), friending requires mutual acceptance and validation through post-friending interactions, creating significant delays in feedback loops. Training generalizable models (id features) with delayed data leads to wide train-eval gap.
- **Privacy constraints**: Limited access to user behavior data
- **Cold start**: Most of the DAU (Daily Active Users) incrementality of friending recommendations is to new users or users who are still building their friending graphs

This project demonstrates a **progressive approach** to friend recommendation, starting with familiar two-tower retrieval models and evolving toward sophisticated generative techniques.

## The Learning Progression

### Phase 1: Traditional Two-Tower Retrieval (Current)

We begin with the "first tech stack you should build today for personalized recommendations" - [two-tower models](https://recsysml.substack.com/p/two-tower-models-for-retrieval-of). This familiar foundation allows us to:

- **Learn user embeddings** from interaction histories
- **Learn target embeddings** from user characteristics
- **Use dot product similarity** for retrieval
- **Build on proven, scalable approaches** used by major tech companies

**Implementation Journey:**
1. **Basic Two-Tower**: Pooled multi-head attention for user history + simple MLP for user-action interaction
2. **Advanced Retrieval**: Mixed negative sampling, latent cross interactions, Mixture of Experts (MoE)
3. **Temporal Grounding**: Use user history as "next item prediction" to reduce overfitting and prepare for generative approaches

This keeps us firmly in the traditional recommender systems retrieval paradigm while building sophistication incrementally.

### Phase 2: Social Tokenized UserIDs (In Progress)

Building on the temporal aspects learned in Phase 1, we represent each user as a configurable token sequence:
- **Learn user tokenization**: Represent users as `num_codebooks` tokens instead of continuous embeddings
- **STU-based history**: Encode user history in terms of STU tokens rather than user IDs
- **Autoregressive prediction**: Predict target users token-by-token with teacher forcing
- **Training focus**: Currently focused on training without inference capabilities

**Key Innovation**: Instead of learning user embeddings, we learn to predict user token sequences, creating a bridge between traditional retrieval and generative approaches.

### Phase 3: Generative Recommendations with Inference (Planned)

Add inference capabilities to enable true generative recommendations:
- **Beam search**: Implement autoregressive inference for generating user sequences
- **Sampling strategies**: Explore temperature sampling, top-k, nucleus sampling
- **Explainable generation**: Generate natural language explanations for recommendations
- **Production deployment**: Move from training-only to full inference pipeline

**Focus**: The transition from training to inference, enabling the model to generate novel user recommendations rather than just ranking existing candidates.

### Phase 4: In-Model Clustering (Planned)

Address scalability through learned clustering, extending Phase 1's two-tower approach:
- **Learned codebook**: Create `C` cluster embeddings (`C << num_users`)
- **Soft clustering**: For each user embedding `u_emb`, compute dot products with all `C` cluster embeddings
- **Weighted representation**: Apply softmax and compute weighted sum of cluster embeddings
- **Search space reduction**: Dramatically reduce the candidate space from millions to thousands

**Key Insight**: Demonstrate that what really helps recommendation quality is the reduction of search space, not just the model architecture.

**Implementation**:
```python
# In __init__:
self.cluster_embeddings = nn.Embedding(num_clusters, embedding_dim)

# For each user embedding u_emb:
cluster_weights = F.softmax(u_emb @ self.cluster_embeddings.weight.T, dim=-1)
clustered_repr = cluster_weights @ self.cluster_embeddings.weight
```

## Why This Progression Matters

Each phase builds on the previous one and teaches fundamental insights:

1. **Phase 1** teaches us user representation and temporal modeling in the familiar two-tower paradigm
2. **Phase 2** shows how to transition from continuous embeddings to discrete token sequences, bridging retrieval and generation
3. **Phase 3** demonstrates that inference capabilities (beam search, sampling) are what truly enable generative recommendations
4. **Phase 4** reveals that search space reduction through clustering is often more important than model complexity

This approach follows the recommender systems field's evolution: **retrieval → tokenization → inference → clustering**, with each phase building deeper understanding of what actually drives recommendation quality.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test_next_target_prediction_userids.py

# Train basic two-tower model (Phase 1, Step 1)
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_data/train_friend_recommendation.py --dataset regular --history_encoder_type simple_attention --interaction_type mlp

# Train advanced retrieval model (Phase 1, Step 2)
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_data/train_friend_recommendation.py --dataset regular --history_encoder_type transformer --interaction_type moe --num_experts 4

# Train on larger dataset to test scalability
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_data/train_friend_recommendation.py --dataset 4x --interaction_type moe --num_experts 4

# Train STU model (Phase 2)
python tests/test_next_target_prediction_stu.py  # Run STU tests first
# STU training script to be added
```

## Project Structure

```
generative_friending_recommendations/
├── src/next_target_prediction_userids/     # Phase 1 implementation
│   ├── next_target_prediction_userids.py   # Two-tower model with advanced features
│   └── README.md                           # Detailed implementation docs
├── src/next_target_prediction_stu/         # Phase 2 implementation
│   ├── next_target_prediction_stu.py       # STU-based autoregressive model
│   └── README.md                           # STU model documentation
├── test_data/                              # Training data and scripts
│   ├── train_friend_recommendation.py      # Unified training script
│   ├── regular/                            # Regular dataset
│   └── 4x/                                 # 4x larger dataset
├── tests/                                  # Test suite
│   ├── test_next_target_prediction_userids.py  # Phase 1 tests
│   └── test_next_target_prediction_stu.py      # Phase 2 tests
└── README.md                               # This file
```

## Current Status: Phase 1 & 2 Results

### Phase 1: Two-Tower Retrieval (Complete)
Our two-tower implementation achieves strong performance on friend recommendation:

| Model | Test Accuracy | Test MRR | Mean Rank | Parameters |
|-------|---------------|----------|-----------|------------|
| **Basic MLP** | 77.33% | 46.21% | 11.29 | 1.28M |
| **Advanced MoE** | **81.91%** | **59.22%** | **8.40** | 1.95M |

**Key Insights:**
- **MoE clearly outperforms MLP** with +4.58% accuracy and +13.01% MRR
- **Excellent cost-performance trade-off**: 53% more parameters for 28% better MRR
- **Production-ready results**: 81.91% accuracy and 59.22% MRR are excellent for friend recommendation

### Phase 2: Social Tokenized UserIDs (In Progress)
- ✅ **Core architecture implemented**: Parametrized STU model with configurable `num_codebooks`
- ✅ **Autoregressive training**: Teacher forcing with token-by-token prediction
- ✅ **Unified tower design**: Eliminated code duplication with dynamic tower creation
- ✅ **Comprehensive testing**: Full test suite with 9 passing tests
- 🔄 **Training script**: In development
- 🔄 **Inference methods**: Beam search and sampling to be added in Phase 3

## Contributing

We welcome contributions! This project is designed to be educational and progressive:

1. **Start with Phase 1**: Understand two-tower models and retrieval
2. **Experiment with features**: Try different negative sampling, attention mechanisms, etc.
3. **Contribute to Phase 2**: Help develop STU training scripts and inference methods
4. **Prepare for Phase 3**: Think about beam search and sampling strategies
5. **Submit improvements**: Better architectures, more efficient training, etc.

### Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd generative_friending_recommendations
pip install -r requirements.txt

# Run tests
python tests/test_next_target_prediction_userids.py

# Start with basic training
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_data/train_friend_recommendation.py --dataset regular
```

## References

- [Two Tower Models for Retrieval](https://recsysml.substack.com/p/two-tower-models-for-retrieval-of) - The foundation we build upon
- [TensorFlow Recommenders](https://www.tensorflow.org/recommenders) - Production-ready two-tower implementations
- [Matrix Factorization](https://developers.google.com/machine-learning/recommendation/collaborative/matrix) - The lineage that led to two-tower models

---

*This project demonstrates the evolution from traditional recommender systems to generative AI, showing how each step builds understanding and capability for the next phase.*