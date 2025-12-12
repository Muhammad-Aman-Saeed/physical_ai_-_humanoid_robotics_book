"""
Policy gradient controller example
File: examples/chapter-9\policy_gradient_controller.py

This example demonstrates a simple policy gradient algorithm applied
to a robotic control task. It uses a neural network to learn
control policies for a simulated robot.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from typing import List, Tuple


class RobotEnvironment:
    """
    A simple robotic environment for policy gradient learning.
    The task is to control a 1-DOF arm to reach a target angle.
    """
    
    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self.reset()
        
    def reset(self):
        """Reset the environment to a random initial state."""
        # Random starting joint angle between -π and π
        self.current_angle = random.uniform(-np.pi, np.pi)
        # Random target angle between -π and π
        self.target_angle = random.uniform(-np.pi, np.pi)
        self.steps_taken = 0
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Return the current state of the environment."""
        # State: [current_angle, target_angle, angular_velocity]
        # Using sin/cos of angles to handle periodicity
        return np.array([
            np.sin(self.current_angle), 
            np.cos(self.current_angle),
            np.sin(self.target_angle), 
            np.cos(self.target_angle),
            0.0  # Initial angular velocity
        ])
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool]:
        """
        Execute an action in the environment.
        
        Args:
            action: Torque to apply to the joint
            
        Returns:
            next_state, reward, done
        """
        # Apply action and update state
        torque = np.clip(action, -2.0, 2.0)  # Limit torque
        angular_acceleration = torque - 0.1 * self.current_angle - 0.05 * self.get_angular_velocity()
        new_velocity = self.get_angular_velocity() + angular_acceleration * 0.05
        new_angle = self.current_angle + new_velocity * 0.05
        
        self.current_angle = np.clip(new_angle, -np.pi, np.pi)
        
        # Calculate reward
        angle_error = abs(self.current_angle - self.target_angle)
        # Normalize angle error to range [0, π], then invert
        reward = np.exp(-angle_error)  # Higher reward for smaller errors
        
        # Small penalty for using large torques
        reward -= 0.01 * abs(torque)
        
        # Check if episode is done
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps
        
        # Large reward if we reach the target
        if angle_error < 0.1:
            reward += 10
            done = True  # Episode ends when target is reached
        
        return self.get_state(), reward, done
    
    def get_angular_velocity(self):
        """For simplicity, return 0. In a more complex env, track velocity."""
        return 0.0


class PolicyNetwork(nn.Module):
    """
    Neural network for policy gradient algorithm.
    Outputs mean and standard deviation of a Gaussian action distribution.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Output mean of action distribution
        self.action_mean = nn.Linear(32, action_dim)
        
        # Output log standard deviation of action distribution
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.network(state)
        action_mean = self.action_mean(x)
        action_std = torch.exp(self.action_log_std)
        return action_mean, action_std


class PolicyGradientAgent:
    """
    Agent using policy gradient to learn a control policy.
    """
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 3e-4):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def select_action(self, state: np.ndarray) -> Tuple[float, float]:
        """
        Select an action using the current policy.
        
        Returns:
            action, log_probability
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_mean, action_std = self.policy(state_tensor)
        
        # Create distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.item(), log_prob.item()
    
    def update_policy(self, states: List[np.ndarray], actions: List[float], 
                     log_probs: List[float], returns: List[float]):
        """
        Update the policy using policy gradient.
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions).unsqueeze(1)
        log_probs_tensor = torch.FloatTensor(log_probs).unsqueeze(1)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1)
        
        # Get action means and stds
        action_means, action_stds = self.policy(states_tensor)
        dist = torch.distributions.Normal(action_means, action_stds)
        
        # Calculate log probabilities of actions
        log_probs_new = dist.log_prob(actions_tensor).sum(dim=-1, keepdim=True)
        
        # Calculate ratio and surrogate loss
        ratio = torch.exp(log_probs_new - log_probs_tensor)
        surrogate_loss = -ratio * returns_tensor
        
        # Update policy
        self.optimizer.zero_grad()
        surrogate_loss.mean().backward()
        # Clip gradients to prevent large updates
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
        self.optimizer.step()


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Compute discounted returns from rewards.
    """
    returns = []
    R = 0
    for reward in reversed(rewards):
        R = reward + gamma * R
        returns.insert(0, R)
    return returns


def train_agent(episodes: int = 1000, render: bool = False):
    """
    Train the policy gradient agent.
    """
    env = RobotEnvironment()
    agent = PolicyGradientAgent(state_dim=5, action_dim=1)  # 5 state dims, 1 action dim
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        
        # Collect a full episode
        while not done:
            action, log_prob = agent.select_action(state)
            
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
        
        # Calculate returns
        returns = compute_returns(rewards)
        
        # Update policy
        agent.update_policy(states, actions, log_probs, returns)
        
        # Record episode statistics
        episode_rewards.append(sum(rewards))
        episode_lengths.append(len(rewards))
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode + 1}/{episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Avg Length (last 100): {avg_length:.2f}")
    
    return agent, episode_rewards, episode_lengths


def plot_training_results(episode_rewards: List[float], episode_lengths: List[float]):
    """
    Plot training results.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    ax1.plot(episode_rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress: Total Reward per Episode')
    ax1.grid(True)
    
    # Plot lengths
    ax2.plot(episode_lengths)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Training Progress: Episode Length')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def test_trained_agent(agent: PolicyGradientAgent, num_tests: int = 5):
    """
    Test the trained agent on new tasks.
    """
    env = RobotEnvironment()
    test_results = []
    
    for i in range(num_tests):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Store trajectory for visualization
        trajectory = []
        
        while not done and steps < env.max_steps:
            action, _ = agent.select_action(state)
            trajectory.append((env.current_angle, env.target_angle))
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
        
        success = abs(env.current_angle - env.target_angle) < 0.1
        test_results.append({
            'total_reward': total_reward,
            'steps': steps,
            'success': success,
            'final_error': abs(env.current_angle - env.target_angle),
            'trajectory': trajectory
        })
        
        print(f"Test {i+1}: "
              f"Reward={total_reward:.2f}, "
              f"Steps={steps}, "
              f"Success={success}, "
              f"Final Error={abs(env.current_angle - env.target_angle):.3f}")
    
    # Visualize one of the test trajectories
    if test_results:
        trajectory = test_results[0]['trajectory']
        if trajectory:
            angles = [t[0] for t in trajectory]
            targets = [t[1] for t in trajectory]
            
            plt.figure(figsize=(12, 6))
            plt.plot(angles, label='Actual Angle', linewidth=2)
            plt.plot(targets, label='Target Angle', linewidth=2, linestyle='--')
            plt.xlabel('Time Step')
            plt.ylabel('Angle (radians)')
            plt.title('Robot Control Trajectory (Test Episode)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
    
    return test_results


def main():
    print("Policy Gradient Controller for Robotics")
    print("=" * 50)
    
    print("Starting training...")
    agent, rewards, lengths = train_agent(episodes=500)
    
    print("\nTraining completed!")
    
    # Plot results
    plot_training_results(rewards, lengths)
    
    print("\nTesting trained agent on new tasks...")
    test_results = test_trained_agent(agent, num_tests=5)
    
    # Summary
    success_rate = sum(1 for result in test_results if result['success']) / len(test_results)
    avg_reward = np.mean([result['total_reward'] for result in test_results])
    avg_steps = np.mean([result['steps'] for result in test_results])
    
    print(f"\nTest Results Summary:")
    print(f"  Success Rate: {success_rate:.2%}")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Steps: {avg_steps:.2f}")
    
    print(f"\nPolicy Gradient concepts demonstrated:")
    print("- Learning control policies through trial and error")
    print("- Using neural networks as function approximators")
    print("- Balancing exploration and exploitation")
    print("- Reward-based learning for robotic tasks")
    print("- Stochastic policies for continuous action spaces")


if __name__ == "__main__":
    main()