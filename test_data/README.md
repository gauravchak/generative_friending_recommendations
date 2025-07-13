# Social Network Data Generation and Training

This directory contains scripts to generate realistic social network data and train the friend recommendation model.

## Overview

The scripts in this directory simulate a growing social network with realistic patterns:
- **Preferential attachment**: Popular users get more friends
- **Clustering**: Friends of friends are more likely to connect
- **Temporal patterns**: Activity varies over time
- **Multiple action types**: friend_request, friend_accept, message, like, comment

## Files

### 1. `generate_social_network_data.py`
Generates realistic social network data by simulating user interactions over time.

**Features:**
- Simulates 10,000 users over 1 year
- Generates 200,000 social actions
- Creates 50,000 training examples
- Follows realistic social network growth patterns

**Usage:**
```bash
python generate_social_network_data.py
```

**Output files:**
- `network_stats.json`: Network statistics
- `training_examples.json`: Training examples for the model
- `all_actions.pkl`: Raw action data
- `user_histories.pkl`: User interaction histories

### 2. `data_loader.py`
Loads the generated data and creates PyTorch data loaders for training.

**Features:**
- Creates train/validation/test splits
- Handles batching and collation
- Provides data analysis utilities

**Usage:**
```bash
python data_loader.py
```

### 3. `train_with_realistic_data.py`
Trains the friend recommendation model using the generated data.

**Features:**
- Complete training pipeline with validation
- Early stopping and learning rate scheduling
- Model checkpointing
- Performance evaluation

**Usage:**
```bash
python train_with_realistic_data.py
```

## Workflow

### Step 1: Generate Data
```bash
cd test_data
python generate_social_network_data.py
```

This will create realistic social network data with the following characteristics:
- **10,000 users** with varying activity levels
- **200,000 actions** over 1 year
- **Realistic friendship patterns** (clustering, preferential attachment)
- **Multiple action types** with realistic distributions

### Step 2: Analyze Data (Optional)
```bash
python data_loader.py
```

This will show statistics about the generated data:
- Action type distribution
- History length statistics
- User activity patterns
- Target popularity distribution

### Step 3: Train Model
```bash
python train_with_realistic_data.py
```

This will:
- Load the generated data
- Create train/validation/test splits
- Train the model with temporal pretraining
- Evaluate performance on the test set
- Save the best model and training history

## Data Format

### Training Examples
Each training example contains:
```json
{
  "actor_id": 1234,
  "actor_history_actions": [2, 0, 3, 1, 0, ...],  // Action types (padded)
  "actor_history_targets": [567, 890, 234, 456, 0, ...],  // Target user IDs (padded)
  "actor_history_mask": [0, 0, 0, 1, 1, 1, ...],  // Validity mask (1=valid, 0=padding)
  "example_action": 0,  // Current action type
  "example_target": 789,  // Current target user
  "timestamp": "2023-06-15T14:30:00",
  "action_success": true
}
```

### Action Types
- `0`: friend_request
- `1`: friend_accept
- `2`: message
- `3`: like
- `4`: comment

## Model Training

The training script uses:
- **Mixed negative sampling**: In-batch + random negatives
- **Temporal pretraining**: Sequence prefix prediction
- **Latent cross interactions**: Elementwise products for richer representations
- **SwiGLU activation**: Modern activation functions with gating mechanism
- **Early stopping**: Prevents overfitting
- **Learning rate scheduling**: Adaptive learning rate

## Expected Results

With the generated data, you should see:
- **Training accuracy**: 0.15-0.25 (realistic for friend recommendation)
- **MRR**: 0.10-0.20 (Mean Reciprocal Rank)
- **Mean Rank**: 15-25 (average rank of correct prediction)

## Customization

You can modify the data generation parameters in `generate_social_network_data.py`:
- `num_users`: Number of users in the network
- `num_actions`: Number of actions to generate
- `max_history_length`: Maximum length of user interaction history
- `start_date`/`end_date`: Time period for the simulation

## Troubleshooting

### Missing Dependencies
If you get import errors, make sure you have:
```bash
pip install torch numpy
```

### Data Not Found
If the training script can't find the data:
1. Make sure you ran `generate_social_network_data.py` first
2. Check that `training_examples.json` exists in the `test_data/` directory

### Memory Issues
If you run out of memory:
- Reduce `num_users` or `num_actions` in the data generation
- Reduce `batch_size` in the training script
- Use CPU instead of GPU if needed

## Performance Tips

1. **Use GPU**: The training will be much faster on GPU
2. **Adjust batch size**: Larger batches are more efficient but use more memory
3. **Monitor metrics**: Watch validation loss to avoid overfitting
4. **Experiment with hyperparameters**: Try different learning rates, temporal weights, etc. 