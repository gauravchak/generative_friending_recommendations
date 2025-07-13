# 4X Larger Social Network Dataset

This directory contains a 4x larger, more realistic social network dataset for testing the friend recommendation model with better performance characteristics.

## Dataset Overview

- **Users**: 8,000 (4x more than original)
- **Actions**: ~80,000 (4x more than original)
- **Realistic Patterns**: Based on real-world social network behavior

## Key Improvements

### 1. Realistic User Behavior
- **Age-based patterns**: Users tend to friend people of similar ages
- **Geographic clustering**: Users in same regions interact more
- **Interest-based homophily**: Users with similar interests connect more
- **Activity level clustering**: Active users tend to be friends with other active users

### 2. Temporal Patterns
- **Hour-of-day patterns**: More activity during evening hours (8-10 PM)
- **Day-of-week patterns**: More activity on weekends
- **Realistic timing**: Actions happen in bursts with realistic intervals

### 3. Action Type Distribution
- `friend_request`: 35% (most common)
- `friend_accept`: 25% (often follows requests)
- `message`: 20% (communication)
- `like_post`: 15% (engagement)
- `comment`: 5% (less common)

### 4. Power Law Distributions
- **User popularity**: Few very popular users, many less popular (realistic)
- **Activity levels**: Most users are moderately active, few are very active
- **Network effects**: Popular users become more popular over time

## Files

- `generate_realistic_social_network_data.py`: Script to generate the dataset
- `train_with_4x_data.py`: Optimized training script for larger dataset
- `social_network_data_4x.json`: The generated dataset
- `network_stats_4x.json`: Dataset statistics and metadata

## Usage

### 1. Generate the Dataset

```bash
cd test_data_4x
python generate_realistic_social_network_data.py
```

This will create:
- `social_network_data_4x.json` (~10-15 MB)
- `network_stats_4x.json` (metadata)

### 2. Train the Model

```bash
python train_with_4x_data.py
```

### 3. Optimizations for Larger Dataset

The training script includes several optimizations:

- **Larger batch size**: 128 (vs 32 in original)
- **More workers**: 4 parallel data loading workers
- **Larger model**: 256-dim embeddings, 512-dim hidden layers
- **Better hyperparameters**: AdamW optimizer, learning rate scheduling
- **Gradient clipping**: Prevents exploding gradients
- **More random negatives**: 10 per batch for better training
- **Reduced temporal weight**: 0.3 (vs 0.5) to balance losses

## Expected Performance

With the larger dataset and optimizations, you should see:

- **Better convergence**: More stable training due to larger batches
- **Higher accuracy**: More training data leads to better generalization
- **Better MRR**: Improved ranking performance
- **Faster training**: Optimized data loading and larger batches

## Dataset Statistics

After generation, you'll see statistics like:

```
Action type distribution:
  friend_request: 28,000 (35.0%)
  friend_accept: 20,000 (25.0%)
  message: 16,000 (20.0%)
  like_post: 12,000 (15.0%)
  comment: 4,000 (5.0%)

User activity statistics:
  Average actions per user: 10.0
  Median actions per user: 8.0
  Min actions per user: 1
  Max actions per user: 45

Popularity statistics:
  Average popularity: 4.0
  Median popularity: 3.0
  Min popularity: 0
  Max popularity: 28
```

## Comparison with Original Dataset

| Metric | Original | 4X Dataset | Improvement |
|--------|----------|------------|-------------|
| Users | 2,000 | 8,000 | 4x |
| Actions | ~20,000 | ~80,000 | 4x |
| Batch Size | 32 | 128 | 4x |
| Embedding Dim | 128 | 256 | 2x |
| Hidden Dim | 256 | 512 | 2x |
| Training Examples | ~14,000 | ~56,000 | 4x |

## Realistic Features

### 1. Homophily
Users are more likely to friend people who are:
- Similar age (±5 years)
- Same geographic region
- Share common interests
- Similar activity levels

### 2. Temporal Patterns
- **Peak hours**: 8-10 PM (highest activity)
- **Weekend boost**: 15-20% more activity on weekends
- **Realistic intervals**: Actions happen every 5-120 minutes

### 3. Network Effects
- **Popularity feedback**: Popular users become more popular
- **Activity clustering**: Active users attract other active users
- **Reciprocity**: Friend requests often lead to mutual connections

### 4. Power Law Behavior
- **User popularity**: Follows Zipf's law (few very popular, many less popular)
- **Activity distribution**: Most users moderately active, few very active
- **Network growth**: New users tend to connect to popular existing users

## Training Recommendations

1. **Use GPU**: The larger dataset benefits significantly from GPU acceleration
2. **Monitor memory**: Larger batches require more memory
3. **Patience**: Training takes longer but should converge to better results
4. **Validation**: Use validation set to prevent overfitting
5. **Checkpointing**: Save best model based on validation MRR

## Expected Training Time

- **CPU**: ~2-3 hours for 20 epochs
- **MPS (Apple Silicon)**: ~45-60 minutes for 20 epochs
- **CUDA**: ~20-30 minutes for 20 epochs

The larger dataset should provide more stable and better-performing models for friend recommendation tasks. 