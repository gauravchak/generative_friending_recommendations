#!/usr/bin/env python3
"""
Generate realistic social network data for friend recommendation testing.
This script creates 4x larger datasets with more realistic friending patterns.
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import os


class RealisticSocialNetworkGenerator:
    """
    Generates realistic social network data based on real-world friending patterns.
    
    Key realistic features:
    1. Power law distribution for user popularity (few very popular users, many less popular)
    2. Temporal patterns (more activity during certain hours/days)
    3. Homophily (users tend to friend similar users)
    4. Reciprocity (friend requests are often mutual)
    5. Activity clustering (active users tend to be friends with other active users)
    6. Geographic clustering (users in similar locations)
    7. Age-based patterns (similar age groups interact more)
    """
    
    def __init__(self, num_users: int = 8000, num_actions: int = 80000):
        self.num_users = num_users
        self.num_actions = num_actions
        
        # User attributes for realistic patterns
        self.user_ages = {}
        self.user_locations = {}
        self.user_activity_levels = {}
        self.user_interests = {}
        
        # Action types with realistic frequencies
        self.action_types = {
            'friend_request': 0.35,    # Most common
            'friend_accept': 0.25,     # Often follows requests
            'message': 0.20,           # Communication
            'like_post': 0.15,         # Engagement
            'comment': 0.05            # Less common
        }
        
        # Time patterns (hour of day weights)
        self.hour_weights = {
            0: 0.02, 1: 0.01, 2: 0.005, 3: 0.005, 4: 0.005, 5: 0.01,
            6: 0.02, 7: 0.03, 8: 0.05, 9: 0.08, 10: 0.10, 11: 0.12,
            12: 0.15, 13: 0.12, 14: 0.10, 15: 0.08, 16: 0.07, 17: 0.08,
            18: 0.12, 19: 0.15, 20: 0.18, 21: 0.20, 22: 0.15, 23: 0.08
        }
        
        # Day of week weights
        self.day_weights = {
            0: 0.12,  # Monday
            1: 0.13,  # Tuesday
            2: 0.14,  # Wednesday
            3: 0.15,  # Thursday
            4: 0.16,  # Friday
            5: 0.18,  # Saturday
            6: 0.12   # Sunday
        }
    
    def generate_user_attributes(self) -> Dict[int, Dict]:
        """Generate realistic user attributes."""
        users = {}
        
        for user_id in range(self.num_users):
            # Age distribution (18-65, weighted towards younger users)
            age_weights = np.exp(-0.05 * np.arange(18, 66))
            age_weights = age_weights / age_weights.sum()
            age = np.random.choice(range(18, 66), p=age_weights)
            
            # Location (simplified to regions)
            regions = ['northeast', 'southeast', 'midwest', 'west', 'international']
            region_weights = [0.25, 0.25, 0.20, 0.20, 0.10]
            location = np.random.choice(regions, p=region_weights)
            
            # Activity level (power law distribution)
            activity_level = np.random.pareto(2.0) + 1  # 1-10 scale
            
            # Interests (multiple interests per user)
            all_interests = ['sports', 'music', 'movies', 'books', 'travel', 
                           'food', 'technology', 'fashion', 'fitness', 'art',
                           'gaming', 'photography', 'cooking', 'politics', 'science']
            num_interests = np.random.poisson(3) + 1  # 1-6 interests
            interests = list(np.random.choice(all_interests, num_interests, replace=False))
            
            users[user_id] = {
                'age': int(age),
                'location': location,
                'activity_level': float(min(activity_level, 10)),
                'interests': interests,
                'created_date': (datetime.now() - timedelta(days=np.random.randint(30, 1000))).isoformat()
            }
        
        return users
    
    def calculate_friendship_probability(self, user1: Dict, user2: Dict) -> float:
        """Calculate probability of friendship based on user attributes."""
        base_prob = 0.01
        
        # Age similarity
        age_diff = abs(user1['age'] - user2['age'])
        age_factor = np.exp(-age_diff / 10)  # Higher for similar ages
        
        # Location similarity
        location_factor = 2.0 if user1['location'] == user2['location'] else 0.5
        
        # Interest overlap
        common_interests = len(set(user1['interests']) & set(user2['interests']))
        interest_factor = 1.0 + common_interests * 0.3
        
        # Activity level similarity
        activity_diff = abs(user1['activity_level'] - user2['activity_level'])
        activity_factor = np.exp(-activity_diff / 3)
        
        return base_prob * age_factor * location_factor * interest_factor * activity_factor
    
    def generate_realistic_actions(self, users: Dict[int, Dict]) -> List[Dict]:
        """Generate realistic social network actions."""
        actions = []
        
        # Start date (6 months ago)
        start_date = datetime.now() - timedelta(days=180)
        
        # Generate actions with realistic timing
        current_time = start_date
        
        for action_id in range(self.num_actions):
            # Time progression (some actions happen in bursts)
            if random.random() < 0.1:  # 10% chance of time jump
                current_time += timedelta(hours=random.randint(1, 24))
            else:
                current_time += timedelta(minutes=random.randint(5, 120))
            
            # Select action type based on weights
            action_type = np.random.choice(
                list(self.action_types.keys()),
                p=list(self.action_types.values())
            )
            
            # Select actor (weighted by activity level)
            activity_levels = [users[uid]['activity_level'] for uid in range(self.num_users)]
            activity_probs = np.array(activity_levels) / sum(activity_levels)
            actor_id = np.random.choice(range(self.num_users), p=activity_probs)
            
            # Select target based on friendship probability
            if action_type in ['friend_request', 'friend_accept']:
                # For friend actions, use friendship probability
                target_probs = []
                for target_id in range(self.num_users):
                    if target_id != actor_id:
                        prob = self.calculate_friendship_probability(
                            users[actor_id], users[target_id]
                        )
                        target_probs.append(prob)
                    else:
                        target_probs.append(0)
                
                # Normalize probabilities
                target_probs = np.array(target_probs)
                if target_probs.sum() > 0:
                    target_probs = target_probs / target_probs.sum()
                    target_id = np.random.choice(range(self.num_users), p=target_probs)
                else:
                    target_id = random.choice([i for i in range(self.num_users) if i != actor_id])
            else:
                # For other actions, use popularity-based selection
                target_id = random.choice(range(self.num_users))
            
            # Add temporal patterns
            hour = current_time.hour
            day = current_time.weekday()
            
            # Adjust probability based on time patterns
            time_factor = self.hour_weights[hour] * self.day_weights[day]
            
            # Only add action if it passes time-based filtering
            if random.random() < time_factor * 10:  # Scale factor
                actions.append({
                    'action_id': int(action_id),
                    'actor_id': int(actor_id),
                    'target_id': int(target_id),
                    'action_type': action_type,
                    'timestamp': current_time.isoformat()
                })
        
        # Sort by timestamp
        actions.sort(key=lambda x: x['timestamp'])
        
        return actions
    
    def generate_network_data(self) -> Dict:
        """Generate complete social network dataset."""
        print(f"Generating realistic social network data...")
        print(f"Users: {self.num_users:,}")
        print(f"Target actions: {self.num_actions:,}")
        
        # Generate user attributes
        print("Generating user attributes...")
        users = self.generate_user_attributes()
        
        # Generate actions
        print("Generating realistic actions...")
        actions = self.generate_realistic_actions(users)
        
        print(f"Generated {len(actions):,} actions")
        
        # Calculate statistics
        action_types = [a['action_type'] for a in actions]
        action_counts = {}
        for action_type in action_types:
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        print("\nAction type distribution:")
        for action_type, count in action_counts.items():
            percentage = (count / len(actions)) * 100
            print(f"  {action_type}: {count:,} ({percentage:.1f}%)")
        
        # User activity statistics
        user_action_counts = {}
        for action in actions:
            actor_id = action['actor_id']
            user_action_counts[actor_id] = user_action_counts.get(actor_id, 0) + 1
        
        activity_counts = list(user_action_counts.values())
        print(f"\nUser activity statistics:")
        print(f"  Average actions per user: {np.mean(activity_counts):.2f}")
        print(f"  Median actions per user: {np.median(activity_counts):.2f}")
        print(f"  Min actions per user: {min(activity_counts)}")
        print(f"  Max actions per user: {max(activity_counts)}")
        
        # Popularity statistics (how often users are targets)
        target_counts = {}
        for action in actions:
            target_id = action['target_id']
            target_counts[target_id] = target_counts.get(target_id, 0) + 1
        
        popularity_counts = list(target_counts.values())
        print(f"\nPopularity statistics:")
        print(f"  Average popularity: {np.mean(popularity_counts):.2f}")
        print(f"  Median popularity: {np.median(popularity_counts):.2f}")
        print(f"  Min popularity: {min(popularity_counts)}")
        print(f"  Max popularity: {max(popularity_counts)}")
        
        return {
            'users': users,
            'actions': actions,
            'metadata': {
                'num_users': self.num_users,
                'num_actions': len(actions),
                'generation_date': datetime.now().isoformat(),
                'action_type_distribution': action_counts,
                'user_activity_stats': {
                    'mean': float(np.mean(activity_counts)),
                    'median': float(np.median(activity_counts)),
                    'min': int(min(activity_counts)),
                    'max': int(max(activity_counts))
                },
                'popularity_stats': {
                    'mean': float(np.mean(popularity_counts)),
                    'median': float(np.median(popularity_counts)),
                    'min': int(min(popularity_counts)),
                    'max': int(max(popularity_counts))
                }
            }
        }


def main():
    """Generate and save realistic social network data."""
    print("Realistic Social Network Data Generator")
    print("=" * 50)
    
    # Create generator with 4x larger dataset
    generator = RealisticSocialNetworkGenerator(
        num_users=8000,      # 4x more users
        num_actions=80000    # 4x more actions
    )
    
    # Generate data
    network_data = generator.generate_network_data()
    
    # Save to file
    output_file = 'social_network_data_4x.json'
    print(f"\nSaving data to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(network_data, f, indent=2)
    
    print(f"Data saved successfully!")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    # Save metadata separately for quick access
    metadata_file = 'network_stats_4x.json'
    with open(metadata_file, 'w') as f:
        json.dump(network_data['metadata'], f, indent=2)
    
    print(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    main() 