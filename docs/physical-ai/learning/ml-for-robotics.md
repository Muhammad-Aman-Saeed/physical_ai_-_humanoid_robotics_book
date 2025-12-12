---
sidebar_position: 8
---

# Machine Learning for Physical Systems

## Learning Objectives

After reading this chapter, you will be able to:
- Understand how machine learning techniques apply to physical systems
- Implement basic learning from demonstration approaches
- Apply supervised learning for perception tasks in robotics
- Use unsupervised learning for pattern recognition in sensor data

## Prerequisites

Before reading this chapter, you should:
- Have basic knowledge of machine learning concepts (supervised, unsupervised learning)
- Understand fundamental robotics concepts from earlier chapters
- Be familiar with Python programming and basic linear algebra

## Introduction

Machine learning in physical systems, or embodied AI, differs significantly from traditional ML applications. When learning algorithms are applied to physical systems, they must account for real-world dynamics, sensor noise, safety constraints, and the physical embodiment of the system. This chapter explores how machine learning techniques are adapted and applied to robotic systems.

## Learning from Demonstration

Learning from demonstration (LfD), also known as imitation learning, is a technique where robots learn to perform tasks by observing human demonstrations. This approach is particularly effective for complex tasks that are difficult to program explicitly.

### Key Concepts

1. **Behavioral Cloning**: Direct mapping from observations to actions
2. **Inverse Optimal Control**: Learning the reward function from demonstrations
3. **Dagger Algorithm**: Iterative approach to improve cloning from states visited by the policy

### Mathematical Foundation

Let `τ = {s₁, a₁, s₂, a₂, ..., sₜ, aₜ}` represent a trajectory where `sᵢ` are states and `aᵢ` are actions. In behavioral cloning, we learn a policy `π(a|s)` that mimics the demonstrated behavior by minimizing:

```
L(θ) = Σᵢ ‖π_θ(sᵢ) - aᵢ‖²
```

### Implementation Example

```python
import numpy as np
from sklearn.neural_network import MLPRegressor

class LearningFromDemonstration:
    def __init__(self, hidden_layers=(100, 50)):
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=1000)
        self.is_trained = False
        
    def train(self, demonstrations):
        """
        Train the model on demonstrations
        :param demonstrations: List of (state, action) tuples
        :return: Trained model
        """
        states = np.array([demo[0] for demo in demonstrations])
        actions = np.array([demo[1] for demo in demonstrations])
        
        self.model.fit(states, actions)
        self.is_trained = True
        return self.model
    
    def predict_action(self, state):
        """
        Predict action for given state
        :param state: Current state vector
        :return: Predicted action
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict([state])[0]

# Example usage
if __name__ == "__main__":
    # Simulated demonstrations: (joint_angles, desired_torque)
    demonstrations = [
        ([0.1, 0.2, 0.1], [0.5, -0.3, 0.2]),
        ([0.2, 0.3, 0.15], [0.6, -0.25, 0.25]),
        ([0.05, 0.15, 0.05], [0.4, -0.35, 0.15]),
        # Add more demonstration data
    ]
    
    lfd = LearningFromDemonstration()
    trained_model = lfd.train(demonstrations)
    
    # Test with a new state
    new_state = [0.15, 0.25, 0.12]
    predicted_action = lfd.predict_action(new_state)
    print(f"State: {new_state}, Predicted action: {predicted_action}")
```

## Supervised Learning for Perception

Robots often need to interpret sensory data to understand their environment. Supervised learning is commonly used for perception tasks such as object detection, segmentation, and classification.

### Common Perception Tasks

1. **Object Detection**: Identify and locate objects in sensor data
2. **Semantic Segmentation**: Label each pixel with object class
3. **Pose Estimation**: Determine position and orientation of objects

### Example: Simple Object Classification

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ObjectClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.is_trained = False
        
    def extract_features(self, sensor_data):
        """
        Extract relevant features from sensor data
        :param sensor_data: Raw sensor readings
        :return: Feature vector
        """
        # Simple features: mean, std, min, max
        features = [
            np.mean(sensor_data),
            np.std(sensor_data), 
            np.min(sensor_data),
            np.max(sensor_data)
        ]
        return features
    
    def train(self, sensor_readings, labels):
        """
        Train classifier on sensor data
        :param sensor_readings: List of sensor reading arrays
        :param labels: Corresponding object labels
        """
        X = [self.extract_features(reading) for reading in sensor_readings]
        self.model.fit(X, labels)
        self.is_trained = True
        
    def classify(self, sensor_reading):
        """
        Classify object based on sensor reading
        :param sensor_reading: New sensor data
        :return: Predicted label
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        features = self.extract_features(sensor_reading)
        prediction = self.model.predict([features])
        probability = self.model.predict_proba([features])
        
        return prediction[0], probability[0]

# Example usage
if __name__ == "__main__":
    # Simulated sensor data for different objects
    # Each entry is a 10-element sensor array
    cube_data = np.random.normal(0.8, 0.1, (10, 10))  # 10 samples of cube
    sphere_data = np.random.normal(0.5, 0.2, (10, 10))  # 10 samples of sphere
    cylinder_data = np.random.normal(0.6, 0.15, (10, 10))  # 10 samples of cylinder
    
    # Combine data and labels
    X = list(cube_data) + list(sphere_data) + list(cylinder_data)
    y = ['cube'] * 10 + ['sphere'] * 10 + ['cylinder'] * 10
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and train classifier
    classifier = ObjectClassifier()
    classifier.train(X_train, y_train)
    
    # Test classification
    test_sample = X_test[0]
    prediction, probability = classifier.classify(test_sample)
    print(f"Predicted object: {prediction}, Probability: {probability}")
```

## Unsupervised Learning for Pattern Recognition

Unsupervised learning techniques are valuable for discovering patterns in sensor data without labeled training examples. These methods are particularly useful for anomaly detection and clustering similar behaviors.

### Clustering Approaches

1. **K-Means**: Partition data into K clusters
2. **Gaussian Mixture Models**: Probabilistic clustering
3. **DBSCAN**: Density-based clustering

### Example: Clustering Robot States

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class StateClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.is_fitted = False
        
    def fit(self, states):
        """
        Fit clustering model to robot states
        :param states: Array of state vectors
        """
        # Normalize the features
        states_scaled = self.scaler.fit_transform(states)
        
        # Apply K-Means clustering
        self.cluster_labels = self.kmeans.fit_predict(states_scaled)
        self.is_fitted = True
        
    def get_cluster(self, state):
        """
        Get cluster assignment for a state
        :param state: State vector
        :return: Cluster index
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        state_scaled = self.scaler.transform([state])
        cluster = self.kmeans.predict(state_scaled)[0]
        return cluster
    
    def visualize_clusters(self, states):
        """
        Visualize clusters (for 2D projection)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        states_scaled = self.scaler.transform(states)
        plt.scatter(states_scaled[:, 0], states_scaled[:, 1], c=self.cluster_labels, cmap='viridis')
        plt.xlabel('Normalized State Feature 1')
        plt.ylabel('Normalized State Feature 2')
        plt.title('Robot State Clusters')
        plt.colorbar()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Simulated robot states (e.g., joint angles and velocities)
    np.random.seed(42)
    states = np.random.rand(100, 4)  # 100 states, 4 dimensions each
    
    # Add some structure to the data
    states[:30, :] += [1, 0, 0, 0]  # First 30 states form cluster 1
    states[30:60, :] += [0, 1, 0, 0]  # Next 30 states form cluster 2
    
    # Create and fit clustering model
    clustering = StateClustering(n_clusters=3)
    clustering.fit(states)
    
    # Test with a new state
    new_state = [1.1, 0.1, 0.2, 0.1]
    cluster = clustering.get_cluster(new_state)
    print(f"New state {new_state} belongs to cluster: {cluster}")
```

## Deep Learning for Physical Systems

Deep learning has revolutionized physical AI systems, particularly in perception and control. Convolutional Neural Networks (CNNs) excel at processing visual data, while Recurrent Neural Networks (RNNs) can handle temporal sequences.

### Example: Deep Q-Network for Simple Control

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DeepQLearningAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = DQN(state_size, 64, action_size)
        self.target_network = DQN(state_size, 64, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
    def get_action(self, state, epsilon=0.1):
        """
        Get action using epsilon-greedy policy
        :param state: Current state
        :param epsilon: Exploration rate
        :return: Action index
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.max(1)[1].item()

# Example usage
if __name__ == "__main__":
    agent = DeepQLearningAgent(state_size=4, action_size=2)
    
    # Simulated state
    state = [0.1, 0.2, 0.3, 0.4]
    action = agent.get_action(state)
    print(f"Action for state {state}: {action}")
```

## Implementation Guidelines

When implementing ML for physical systems, follow these guidelines:

1. **Safety First**: Always include safety checks when applying learned policies to physical robots
2. **Data Quality**: Clean and verify sensor data before training
3. **Validation**: Test extensively in simulation before physical deployment
4. **Robustness**: Account for sensor noise and environmental variations
5. **Interpretability**: Where possible, maintain model interpretability for debugging

## Summary

This chapter introduced key machine learning techniques for physical systems, including learning from demonstration, supervised learning for perception, unsupervised learning for pattern recognition, and deep learning approaches. Each technique was demonstrated with practical code examples that can be adapted for various robotic applications.

## Exercises

1. Modify the LearningFromDemonstration class to handle continuous action spaces instead of discrete ones
2. Implement a CNN-based object classifier using PyTorch for processing camera images
3. Extend the StateClustering class to automatically determine the optimal number of clusters

## References

- Argall, B. D., Chernova, S., Veloso, M., & Browning, B. (2009). A survey of robot learning from demonstration. *Robotics and autonomous systems*, 57(5), 469-483.
- Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement learning in robotics: A survey. *The International Journal of Robotics Research*, 32(11), 1238-1274.
- Levine, S., Pastor, P., Krizhevsky, A., & Quillen, D. (2016). Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection. *The International Journal of Robotics Research*, 37(4-5), 421-436.