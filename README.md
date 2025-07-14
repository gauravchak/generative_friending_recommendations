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

### Phase 2: Social Tokenized UserIDs (Planned)

Building on the temporal aspects learned in Phase 1, we'll represent each user as a 3-token sequence:
- Change from target prediction to next-token prediction
- Enable generative capabilities
- Maintain the user history modeling we've developed

### Phase 3: Generative Recommendations (Planned)

Use language models to generate friend suggestions:
- Leverage tokenized user representations for natural language output
- Enable explainable recommendations ("I'm suggesting Sarah because you both enjoy hiking and live in the same neighborhood")
- Move beyond retrieval to true generation

### Phase 4: In-Model Clustering (Planned)

Address scalability challenges:
- Intermediate clustering to reduce embedding table size
- Maintain performance while reducing complexity
- Handle large user bases efficiently

## Why This Progression Matters

Each phase builds on the previous one:

1. **Phase 1** teaches us user representation and temporal modeling
2. **Phase 2** transitions from retrieval to generation while keeping user context
3. **Phase 3** enables natural language explanations and more nuanced recommendations
4. **Phase 4** ensures the system scales to real-world usage

This approach follows the recommender systems field's evolution: **retrieval → ranking → generation**.

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
```

## Project Structure

```
generative_friending_recommendations/
├── src/next_target_prediction_userids/     # Phase 1 implementation
│   ├── next_target_prediction_userids.py   # Two-tower model with advanced features
│   └── README.md                           # Detailed implementation docs
├── test_data/                              # Training data and scripts
│   ├── train_friend_recommendation.py      # Unified training script
│   ├── regular/                            # Regular dataset
│   └── 4x/                                 # 4x larger dataset
├── tests/                                  # Test suite
└── README.md                               # This file
```

## Current Status: Phase 1 Results

Our two-tower implementation achieves strong performance on friend recommendation:

| Model | Test Accuracy | Test MRR | Mean Rank | Parameters |
|-------|---------------|----------|-----------|------------|
| **Basic MLP** | 77.33% | 46.21% | 11.29 | 1.28M |
| **Advanced MoE** | **81.91%** | **59.22%** | **8.40** | 1.95M |

**Key Insights:**
- **MoE clearly outperforms MLP** with +4.58% accuracy and +13.01% MRR
- **Excellent cost-performance trade-off**: 53% more parameters for 28% better MRR
- **Production-ready results**: 81.91% accuracy and 59.22% MRR are excellent for friend recommendation

## Contributing

We welcome contributions! This project is designed to be educational and progressive:

1. **Start with Phase 1**: Understand two-tower models and retrieval
2. **Experiment with features**: Try different negative sampling, attention mechanisms, etc.
3. **Prepare for Phase 2**: Think about how to tokenize user representations
4. **Submit improvements**: Better architectures, more efficient training, etc.

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