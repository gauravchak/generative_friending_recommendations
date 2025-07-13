#!/usr/bin/env python3
"""
Generate realistic social network data for testing friend recommendation models.

This script simulates a growing social network by:
1. Starting with a set of users
2. Simulating friending actions over time
3. Collecting interaction histories for each user
4. Creating training examples for next-target prediction

The generated data follows realistic social network patterns:
- Preferential attachment (popular users get more friends)
- Clustering (friends of friends are more likely to connect)
- Temporal patterns (activity varies over time)
- Different types of social actions
"""

import json
from datetime import datetime, timedelta
import random


def generate_social_network_data(
    num_users=100,  # Reduced from 10,000
    num_actions=1000,  # Reduced from 50,000
    start_date="2023-01-01",
    end_date="2024-01-01",
    output_file="social_network_data.json"
):
    """
    Generate realistic social network data for testing friend recommendation models.
    
    Args:
        num_users: Number of users in the network
        num_actions: Number of actions to generate
        start_date: Start date for the simulation
        end_date: End date for the simulation
        output_file: Output file path
    """
    print(f"Generating social network data with {num_users} users and {num_actions} actions...")
    
    # Initialize users
    users = list(range(1, num_users + 1))
    user_creation_dates = {
        user_id: datetime.strptime(start_date, "%Y-%m-%d") + timedelta(
            days=random.randint(0, 30)  # Users join within first month
        )
        for user_id in users
    }
    
    # Initialize user popularity (number of followers)
    user_popularity = {user_id: 0 for user_id in users}
    
    # Initialize user clusters (simulate communities)
    num_clusters = max(1, num_users // 20)  # ~5 users per cluster
    user_clusters = {}
    for user_id in users:
        cluster_id = random.randint(1, num_clusters)
        user_clusters[user_id] = cluster_id
    
    # Generate actions
    actions = []
    date_range = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
    
    # Action types and their probabilities
    action_types = ["follow", "like", "comment", "share"]
    action_weights = [0.4, 0.3, 0.2, 0.1]  # Follow is most common
    
    # Track existing connections to avoid duplicates
    existing_connections = set()
    
    for i in range(num_actions):
        # Actor: choose based on popularity (preferential attachment)
        actor_weights = [user_popularity.get(user_id, 0) + 1 for user_id in users]
        actor = random.choices(users, weights=actor_weights)[0]
        
        # Target: choose based on popularity and cluster similarity
        target_candidates = [u for u in users if u != actor]
        
        # Calculate target weights based on popularity and cluster similarity
        target_weights = []
        for target in target_candidates:
            # Base popularity weight
            popularity_weight = user_popularity.get(target, 0) + 1
            
            # Cluster similarity bonus (higher weight for same cluster)
            cluster_bonus = 2.0 if user_clusters.get(actor) == user_clusters.get(target) else 1.0
            
            # Avoid duplicate connections
            connection_penalty = 0.1 if (actor, target) in existing_connections else 1.0
            
            target_weights.append(popularity_weight * cluster_bonus * connection_penalty)
        
        target = random.choices(target_candidates, weights=target_weights)[0]
        
        # Choose action type
        action_type = random.choices(action_types, weights=action_weights)[0]
        
        # Generate timestamp (more recent actions are more likely)
        days_from_start = random.randint(0, date_range)
        # Add some clustering in time (actions happen in bursts)
        if random.random() < 0.3:  # 30% chance to cluster with recent actions
            if actions:
                recent_days = actions[-1]["timestamp"].split("T")[0]
                recent_date = datetime.strptime(recent_days, "%Y-%m-%d")
                days_from_start = min(date_range, max(0, (recent_date - datetime.strptime(start_date, "%Y-%m-%d")).days + random.randint(-2, 2)))
        
        timestamp = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=days_from_start)
        timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Ensure actor exists before target
        if timestamp < user_creation_dates[actor]:
            timestamp = user_creation_dates[actor] + timedelta(hours=random.randint(1, 24))
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        
        if timestamp < user_creation_dates[target]:
            timestamp = user_creation_dates[target] + timedelta(hours=random.randint(1, 24))
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        
        action = {
            "actor_id": actor,
            "target_id": target,
            "action_type": action_type,
            "timestamp": timestamp_str
        }
        
        actions.append(action)
        
        # Update popularity for follow actions
        if action_type == "follow":
            user_popularity[target] = user_popularity.get(target, 0) + 1
            existing_connections.add((actor, target))
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_actions} actions...")
    
    # Sort actions by timestamp
    print("Sorting actions by timestamp...")
    actions.sort(key=lambda x: x["timestamp"])
    
    # Create final dataset
    dataset = {
        "metadata": {
            "num_users": num_users,
            "num_actions": len(actions),
            "date_range": f"{start_date} to {end_date}",
            "generated_at": datetime.now().isoformat()
        },
        "users": {
            str(user_id): {
                "creation_date": user_creation_dates[user_id].strftime("%Y-%m-%d"),
                "cluster": user_clusters[user_id],
                "final_popularity": user_popularity[user_id]
            }
            for user_id in users
        },
        "actions": actions
    }
    
    # Save to file
    print(f"Saving data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Generated {len(actions)} actions for {num_users} users")
    print(f"📊 Popularity distribution: min={min(user_popularity.values())}, max={max(user_popularity.values())}, avg={sum(user_popularity.values())/len(user_popularity):.1f}")
    print(f"📁 Data saved to {output_file}")
    
    return dataset


def main():
    """Generate social network data."""
    print("Social Network Data Generator")
    print("=" * 40)
    
    # Generate larger dataset for better training
    dataset = generate_social_network_data(
        num_users=2000,  # More users for larger dataset
        num_actions=20000,  # 20K actions for better training
        start_date="2023-01-01",
        end_date="2024-01-01",
        output_file="social_network_data.json"
    )
    
    print("\nData generation completed!")
    print("Files created:")
    print("- social_network_data.json: Complete dataset with users and actions")


if __name__ == "__main__":
    main() 