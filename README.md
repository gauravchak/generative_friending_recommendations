# generative_friending_recommendations
This repository walks through a few ideas of how to implement generative recommendations for friending recommendations and in doing so describes the complexity or what is uniquely difficult in friending versus content recommendations.

## Next target prediction using UserIDs (Pytorch implementation)

This part is not technically generative recommendations and can be characterized better as next target prediction. We see this as a stepping stone towards explaining generative recommendations. Our motivation is that readers who are more conversant with recsys applications like two tower models will feel at home in this implementation. Then when we move to the next section generative recommendations, it will be a smaller step.

**📁 Implementation**: [`src/next_target_prediction_userids/`](src/next_target_prediction_userids/)

**📖 Documentation**: See the [detailed README](src/next_target_prediction_userids/README.md) for comprehensive implementation details, usage examples, and model architecture.

**🔧 Key Features**:
- Transformer-based history encoder for user interaction sequences
- **Two training approaches**: Random negative sampling and in-batch negative sampling (IBN)
- Variable-length history handling with attention masks
- Ranking metrics (accuracy, mean rank, MRR)
- Complete training and inference pipeline
- Actor-action representation for efficient retrieval

**🚀 Quick Start**:
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test_next_target_prediction_userids.py

# Run example
cd src/next_target_prediction_userids
python example_usage.py
```

The implementation provides a solid foundation for the subsequent sections on Social Tokenized UserIDs and generative recommendations.

## Next target prediction using Social Tokenized UserIds (Pytorch implementation)
This will assume the existence of a Social Tokenized UserID (STU) a 3 token UserID for each user. This will change training from in batch negatives to next token prediction. Just to keep the leap short this section will only focus on training.

## Inference of next target prediction using STU (Pytorch implementation)
Here we learn to add inference for Social Tokenized UserID (STU) based prediction. The model trained on UserIDs is inferred using k-nearest-neighbors.

## In model clustering as an intermediate step
Here we pose the insight to the reader that in the context of friending recommendations, an intermediate step that might still provide a lot of value is the ability to train a clustering step during next target prediction and use that for next token prediction. Since the clusters have a lower cardinality, the resulting cluster embedding table is smaller, and training a loss from that will lead to less overfitting concerns.