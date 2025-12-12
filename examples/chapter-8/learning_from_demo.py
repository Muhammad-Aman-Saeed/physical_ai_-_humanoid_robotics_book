"""
Learning from demonstration example
File: examples/chapter-8\learning_from_demo.py

This example demonstrates learning from demonstration (LfD) using
supervised learning to teach a robot a trajectory following task.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math


class RobotLfD:
    """
    Learning from Demonstration (LfD) system for teaching robot trajectories.
    """
    
    def __init__(self, n_features=4, n_outputs=2):
        """
        Initialize the LfD system.
        
        Args:
            n_features: Number of input features (e.g., current x, y, target x, y)
            n_outputs: Number of outputs (e.g., next x, y position)
        """
        self.n_features = n_features
        self.n_outputs = n_outputs
        
        # Neural network to learn the demonstration
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 50),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        
        # Scaler for normalizing inputs
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Storage for demonstrations
        self.demonstrations = []
        
    def generate_demonstration_trajectory(self, start_pos=(0, 0), end_pos=(5, 5), n_points=20):
        """
        Generate a demonstration trajectory (e.g., a line, circle, or other shape).
        """
        trajectory = []
        
        # Create a curved trajectory (e.g., cubic Bezier curve or modified straight line)
        for i in range(n_points):
            t = i / (n_points - 1)  # normalized parameter from 0 to 1
            
            # Create a smooth curve from start to end with some curvature
            # This creates a trajectory that's not just a straight line
            x = start_pos[0] + (end_pos[0] - start_pos[0]) * t
            y = start_pos[1] + (end_pos[1] - start_pos[1]) * t + 1.0 * math.sin(t * math.pi)
            
            trajectory.append([x, y])
        
        return np.array(trajectory)
    
    def create_training_data(self, trajectory):
        """
        Create training data from a demonstration trajectory.
        Each input is the current position + target, output is the next position.
        """
        X = []
        y = []
        
        for i in range(len(trajectory) - 1):
            # Input features: current x, current y, end x, end y
            current_pos = trajectory[i]
            end_pos = trajectory[-1]  # Use the final position as target
            X.append([current_pos[0], current_pos[1], end_pos[0], end_pos[1]])
            
            # Output: next position
            next_pos = trajectory[i + 1]
            y.append([next_pos[0], next_pos[1]])
        
        return np.array(X), np.array(y)
    
    def add_demonstration(self, trajectory):
        """
        Add a demonstration trajectory to the system.
        """
        X, y = self.create_training_data(trajectory)
        self.demonstrations.append((X, y))
        return X, y
    
    def train(self):
        """
        Train the model on all demonstrations.
        """
        if not self.demonstrations:
            raise ValueError("No demonstrations available for training")
        
        # Combine all demonstration data
        all_X = []
        all_y = []
        
        for X, y in self.demonstrations:
            all_X.append(X)
            all_y.append(y)
        
        X_combined = np.vstack(all_X)
        y_combined = np.vstack(all_y)
        
        # Scale the features
        X_scaled = self.scaler_X.fit_transform(X_combined)
        y_scaled = self.scaler_y.fit_transform(y_combined)
        
        # Train the model
        self.model.fit(X_scaled, y_scaled.ravel() if self.n_outputs == 1 else y_scaled)
        
        print(f"Trained model on {len(X_combined)} data points")
        
    def predict_next_position(self, current_pos, target_pos):
        """
        Predict the next position given current position and target.
        """
        # Create input vector
        input_vector = np.array([current_pos[0], current_pos[1], target_pos[0], target_pos[1]]).reshape(1, -1)
        
        # Scale the input
        input_scaled = self.scaler_X.transform(input_vector)
        
        # Make prediction
        prediction_scaled = self.model.predict(input_scaled)
        
        # Reshape if needed for inverse transform
        if self.n_outputs == 1:
            prediction_scaled = prediction_scaled.reshape(-1, 1)
        else:
            prediction_scaled = prediction_scaled.reshape(1, -1)
        
        # Inverse scale to get real-world coordinates
        prediction = self.scaler_y.inverse_transform(prediction_scaled)
        
        return prediction.flatten()
    
    def execute_trajectory(self, start_pos, target_pos, n_steps=20):
        """
        Execute a learned trajectory from start to target position.
        """
        trajectory = [start_pos]
        current_pos = start_pos
        
        for _ in range(n_steps):
            next_pos = self.predict_next_position(current_pos, target_pos)
            trajectory.append(next_pos)
            current_pos = next_pos
            
            # Check if we're close enough to the target
            if np.linalg.norm(np.array(current_pos) - np.array(target_pos)) < 0.1:
                break
        
        return np.array(trajectory)


def main():
    print("Learning from Demonstration (LfD) Example")
    print("=" * 50)
    
    # Create the LfD system
    lfd = RobotLfD()
    
    # Generate multiple demonstrations for different start/end points
    print("Generating demonstrations...")
    
    # Demonstration 1: from (0,0) to (5,5)
    demo1 = lfd.generate_demonstration_trajectory(start_pos=(0, 0), end_pos=(5, 5))
    
    # Demonstration 2: from (1,1) to (6,4)
    demo2 = lfd.generate_demonstration_trajectory(start_pos=(1, 1), end_pos=(6, 4))
    
    # Demonstration 3: from (0,2) to (4,6)
    demo3 = lfd.generate_demonstration_trajectory(start_pos=(0, 2), end_pos=(4, 6))
    
    # Add demonstrations to the system
    lfd.add_demonstration(demo1)
    lfd.add_demonstration(demo2)
    lfd.add_demonstration(demo3)
    
    print(f"Added {len(lfd.demonstrations)} demonstrations")
    
    # Train the model
    print("Training the model...")
    lfd.train()
    
    # Test the learned behavior
    print("\nTesting learned trajectories...")
    
    # Test trajectory 1: from (0,0) to (5,5) - similar to training
    test_traj1 = lfd.execute_trajectory(start_pos=(0, 0), target_pos=(5, 5))
    
    # Test trajectory 2: from (2,1) to (7,5) - extrapolation
    test_traj2 = lfd.execute_trajectory(start_pos=(2, 1), target_pos=(7, 5))
    
    # Test trajectory 3: from (1,3) to (3,7) - different from training
    test_traj3 = lfd.execute_trajectory(start_pos=(1, 3), target_pos=(3, 7))
    
    print("Testing completed!")
    
    # Visualization
    print("\nGenerating visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot demonstrations
    colors = ['red', 'green', 'blue']
    labels = ['Demo 1', 'Demo 2', 'Demo 3']
    
    for i, demo in enumerate([demo1, demo2, demo3]):
        plt.plot(demo[:, 0], demo[:, 1], '--', color=colors[i], linewidth=2, label=f'Demonstration {labels[i]}')
        plt.scatter(demo[0, 0], demo[0, 1], s=100, color=colors[i], marker='o', zorder=5)
        plt.scatter(demo[-1, 0], demo[-1, 1], s=100, color=colors[i], marker='s', zorder=5)
    
    # Plot learned trajectories
    test_colors = ['orange', 'purple', 'brown']
    test_labels = ['Learned 1', 'Learned 2', 'Learned 3']
    
    for i, test_traj in enumerate([test_traj1, test_traj2, test_traj3]):
        plt.plot(test_traj[:, 0], test_traj[:, 1], '-', color=test_colors[i], linewidth=2, 
                 label=f'Learned {test_labels[i]}', alpha=0.7)
        plt.scatter(test_traj[0, 0], test_traj[0, 1], s=100, color=test_colors[i], marker='^', zorder=5)
        plt.scatter(test_traj[-1, 0], test_traj[-1, 1], s=100, color=test_colors[i], marker='D', zorder=5)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Learning from Demonstration: Robot Trajectory Learning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
    
    # Print stats
    print(f"\nDemonstration trajectories:")
    print(f"  Demo 1: {len(demo1)} points from (0,0) to (5,5)")
    print(f"  Demo 2: {len(demo2)} points from (1,1) to (6,4)")
    print(f"  Demo 3: {len(demo3)} points from (0,2) to (4,6)")
    
    print(f"\nLearned trajectories:")
    print(f"  Learned 1: {len(test_traj1)} points from (0,0) to (5,5)")
    print(f"  Learned 2: {len(test_traj2)} points from (2,1) to (7,5)")
    print(f"  Learned 3: {len(test_traj3)} points from (1,3) to (3,7)")
    
    print(f"\nLearning from Demonstration (LfD) concepts demonstrated:")
    print("- Teaching robot behaviors through human demonstration")
    print("- Generalizing from examples to new situations")
    print("- Mapping current state + goal to next action")
    print("- Neural network learning of trajectory patterns")


if __name__ == "__main__":
    main()