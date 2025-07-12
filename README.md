# generative_friending_recommendations
This repository walks through a few ideas of how to implement generative recommendations for friending recommendations and in doing so describes the complexity or what is uniquely difficult in friending versus content recommendations.

## Next target prediction using UserIDs (Pytorch implementation)
This part is not technically generative recommendations and can be characterized better as next target prediction. We see this as a stepping stone towards explaining generative recommendations. Our motivation is that readers who are more conversant with recsys applications like two tower models will feel at home in this implementation. Then when we move to the next section generative recommendations, it will be a smaller step.

## Next target prediction using Social Tokenized UserIds (Pytorch implementation)
This will assume the existence of a Social Tokenized UserID (STU) a 3 token UserID for each user. This will change training from in batch negatives to next token prediction. Just to keep the leap short this section will only focus on training.

## Inference of next target prediction using STU (Pytorch implementation)
Here we learn to add inference for Social Tokenized UserID (STU) based prediction. The model trained on UserIDs is inferred using K-nearest-neighbors.

## In model clusrtering as an intermediate step
