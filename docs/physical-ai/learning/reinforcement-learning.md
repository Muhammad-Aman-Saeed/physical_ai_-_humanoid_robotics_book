---
sidebar_position: 9
---

# Reinforcement Learning in Robotics

## Learning Objectives

After reading this chapter, you will be able to:
- Understand Markov Decision Processes in the context of robotics
- Implement policy gradient methods for robot control
- Apply Deep Reinforcement Learning for robotic control tasks
- Design reward functions for robotic tasks

## Prerequisites

Before reading this chapter, you should:
- Have knowledge of basic machine learning concepts
- Understand fundamental robotics concepts from earlier chapters
- Be familiar with probability theory and optimization
- Have completed the previous chapter on Machine Learning for Physical Systems

## Introduction

Reinforcement Learning (RL) has emerged as a powerful paradigm for learning control policies in robotics. Unlike supervised learning, RL allows robots to learn optimal behaviors through trial and error, receiving rewards for successful actions and penalties for failures. This chapter explores how RL techniques are applied to robotic systems, from simple control tasks to complex manipulation and navigation.

## Markov Decision Processes in Robotics

A Markov Decision Process (MDP) provides the mathematical framework for RL problems. In robotics, an MDP is defined by:

- **States (S)**: The robot's configuration, joint angles, sensor readings
- **Actions (A)**: Motor commands, joint torques, high-level commands
- **Transition probabilities (P)**: How the robot moves between states given actions
- **Rewards (R)**: Feedback signal for goal achievement
- **Discount factor (γ)**: Preference for immediate vs. future rewards

### Mathematical Representation

An MDP is formally defined as a tuple ⟨S, A, P, R, γ⟩ where:
- <code>s<sub>t</sub> ∈ S</code> represents the state at time t
- <code>a<sub>t</sub> ∈ A</code> represents the action taken at time t
- <code>P(s<sub>t+1</sub>|s<sub>t</sub>, a<sub>t</sub>)</code> is the probability of transitioning to state s<sub>t+1</sub> after taking action a<sub>t</sub> in state s<sub>t</sub>
- <code>R(s<sub>t</sub>, a<sub>t</sub>, s<sub>t+1</sub>)</code> is the reward received after the transition

### Example: Simple Navigation MDP

```python
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class RobotMDP:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.num_states = grid_size * grid_size
        self.num_actions = 4
        self.goal_state = self._state_from_pos(grid_size-1, grid_size-1)
        self.obstacles = [(1, 1), (2, 1), (3, 1)]  # Obstacle positions
        
    def _state_from_pos(self, x, y):
        """Convert (x,y) coordinates to state index"""
        return x * self.grid_size + y
    
    def _pos_from_state(self, state):
        """Convert state index to (x,y) coordinates"""
        x = state // self.grid_size
        y = state % self.grid_size
        return x, y
    
    def reward(self, state, action, next_state):
        """Calculate reward for transition"""
        x, y = self._pos_from_state(next_state)
        
        # Check if it's an obstacle
        if (x, y) in self.obstacles:
            return -10  # Penalty for hitting obstacle
        
        # Check if it's the goal
        if next_state == self.goal_state:
            return 10  # Reward for reaching goal
        
        # Small penalty for each step (to encourage efficiency)
        return -1
    
    def transition(self, state, action):
        """Get possible next states and their probabilities"""
        x, y = self._pos_from_state(state)
        
        # Determine intended next position based on action
        if action == Action.UP.value:
            next_x, next_y = max(0, x-1), y
        elif action == Action.DOWN.value:
            next_x, next_y = min(self.grid_size-1, x+1), y
        elif action == Action.LEFT.value:
            next_x, next_y = x, max(0, y-1)
        elif action == Action.RIGHT.value:
            next_x, next_y = x, min(self.grid_size-1, y+1)
        
        # In a deterministic environment, probability is 1.0 for intended state
        next_state = self._state_from_pos(next_x, next_y)
        return [(1.0, next_state)]  # (probability, state)

# Example usage
mdp = RobotMDP(grid_size=5)
print(f"Grid size: {mdp.grid_size}x{mdp.grid_size}")
print(f"Goal state: {mdp.goal_state}")
print(f"State (0,0) corresponds to index: {mdp._state_from_pos(0, 0)}")
```

## Policy Gradient Methods

Policy gradient methods directly optimize the policy function π(a|s) that maps states to actions, rather than learning a value function first. This approach is particularly useful in robotics where action spaces might be continuous.

### REINFORCE Algorithm

The REINFORCE algorithm updates the policy parameters θ in the direction of the gradient:

```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) G_t]
```

Where G_t is the total discounted reward from time t.

### Example: REINFORCE for Robot Control

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, state):
        logits = self.network(state)
        return torch.softmax(logits, dim=-1)

class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        """Select an action based on current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy(state_tensor)
        
        # Sample action from the probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        self.log_probs.append(log_prob)
        return action.item()
    
    def store_reward(self, reward):
        """Store reward received after taking an action"""
        self.rewards.append(reward)
    
    def update_policy(self):
        """Update the policy based on collected rewards"""
        # Compute discounted returns
        returns = []
        R = 0
        for reward in self.rewards[::-1]:
            R = reward + 0.99 * R  # Discount factor = 0.99
            returns.insert(0, R)
        
        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Reset storage
        self.log_probs = []
        self.rewards = []

# Example: Simulated robot environment
class SimpleRobotEnv:
    def __init__(self):
        self.state = np.random.uniform(-1, 1, size=(4,))  # 4-dimensional state
        self.goal = np.array([0.0, 0.0, 0.0, 0.0])
    
    def step(self, action):
        """Simulate robot taking an action"""
        # Update state based on action (simplified physics)
        self.state = self.state + np.random.uniform(-0.1, 0.1, size=self.state.shape)
        self.state[0] = np.clip(self.state[0], -1, 1)  # Keep within bounds
        
        # Calculate reward (negative distance to goal)
        distance_to_goal = np.linalg.norm(self.state - self.goal)
        reward = -distance_to_goal
        
        # Determine if episode is done
        done = distance_to_goal < 0.1  # Close enough to goal
        return self.state.copy(), reward, done

# Example usage
if __name__ == "__main__":
    env = SimpleRobotEnv()
    agent = REINFORCEAgent(state_size=4, action_size=2)  # 2 discrete actions
    
    num_episodes = 1000
    
    for episode in range(num_episodes):
        state = env.state
        total_reward = 0
        done = False
        
        # Run one episode
        while not done:
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            agent.store_reward(reward)
            total_reward += reward
        
        # Update policy after episode
        agent.update_policy()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
        
        # Reset environment for next episode
        env = SimpleRobotEnv()
```

## Deep Deterministic Policy Gradient (DDPG)

DDPG is an actor-critic method designed for continuous action spaces, which is common in robotics. It combines the benefits of value-based and policy-based methods.

### Actor-Critic Architecture

- **Actor**: Learns the policy π(s) that maps states to actions
- **Critic**: Learns the action-value function Q(s, a)

### Example: DDPG Implementation for Robotic Control

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh()  # Actions are bounded between -1 and 1
        )
    
    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.state_input = nn.Linear(state_size, hidden_size)
        self.action_input = nn.Linear(action_size, hidden_size)
        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state, action):
        state_out = self.state_input(state)
        action_out = self.action_input(action)
        combined = torch.cat([state_out, action_out], dim=1)
        return self.network(combined)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, state_size, action_size, lr_actor=1e-4, lr_critic=1e-3, tau=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau  # Soft update parameter
        
        # Initialize networks
        self.actor = Actor(state_size, action_size)
        self.actor_target = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.critic_target = Critic(state_size, action_size)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize target networks with same weights
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # Experience replay buffer
        self.memory = ReplayBuffer()
        
    def hard_update(self, target, source):
        """Hard update target network with source network weights"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source):
        """Soft update target network with source network weights"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def act(self, state, noise=None):
        """Select action based on current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()
        if noise is not None:
            action += noise
        return np.clip(action, -1, 1)  # Clip to action bounds
    
    def update(self, batch_size=64, gamma=0.99):
        """Update networks based on experiences from buffer"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from buffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Compute target Q-values
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions.detach())
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
        
        # Update critic
        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

# Example usage with a simple robot simulation
class SimpleManipulatorEnv:
    def __init__(self):
        # Robot has 3 joints, task is to reach a target position
        self.state = np.random.uniform(-1, 1, size=(6,))  # 3 joint angles + 3 joint velocities
        self.target = np.array([0.5, 0.3, -0.2])  # Target position (x, y, z)
        self.steps = 0
        self.max_steps = 100
    
    def step(self, action):
        """Simulate robot taking an action"""
        # Simplified dynamics: update joint angles based on action
        self.state[:3] = self.state[:3] + action * 0.1  # Scale action effect
        self.state[3:] = np.random.uniform(-0.1, 0.1, size=3)  # Simulated joint velocities
        
        # Calculate reward (negative distance to target)
        end_effector_pos = np.sin(self.state[:3])  # Simplified forward kinematics
        distance_to_target = np.linalg.norm(end_effector_pos - self.target)
        
        # Reward is negative distance (higher reward = closer to target)
        reward = -distance_to_target
        
        self.steps += 1
        done = self.steps >= self.max_steps or distance_to_target < 0.1  # Success condition
        
        return self.state.copy(), reward, done, {}

# Example usage
if __name__ == "__main__":
    env = SimpleManipulatorEnv()
    agent = DDPGAgent(state_size=6, action_size=3)  # 3 joint control actions
    
    num_episodes = 500
    scores = []
    
    for episode in range(num_episodes):
        state = env.state
        total_reward = 0
        done = False
        
        while not done:
            # Add noise for exploration
            noise = np.random.normal(0, 0.1, size=3)
            action = agent.act(state, noise)
            
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # Update agent from experience replay
            agent.update()
        
        scores.append(total_reward)
        
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
        
        # Reset environment for next episode
        env = SimpleManipulatorEnv()
```

## Safety Considerations in RL for Robotics

When applying RL to physical robots, safety is paramount. Here are key considerations:

1. **Constraint Handling**: Incorporate safety constraints directly into the learning algorithm
2. **Safe Exploration**: Use techniques like model-based RL to predict potentially dangerous states
3. **Robust Reward Design**: Ensure reward functions don't incentivize unsafe behaviors
4. **Physical Limits**: Respect joint limits, velocity limits, and other physical constraints

### Example: Safe Exploration with Action Constraints

```python
class ConstrainedDDPGAgent(DDPGAgent):
    def __init__(self, state_size, action_size, joint_limits=None, velocity_limits=None):
        super().__init__(state_size, action_size)
        self.joint_limits = joint_limits if joint_limits else [(-1, 1)] * action_size
        self.velocity_limits = velocity_limits if velocity_limits else [(-0.5, 0.5)] * action_size
    
    def act_with_constraints(self, state, current_joints, noise=None):
        """Select safe action respecting physical limits"""
        # Get base action from policy
        raw_action = self.act(state, noise)
        
        # Apply joint limit constraints
        new_joints = current_joints + raw_action * 0.1  # Scale action appropriately
        
        safe_actions = []
        for i, (joint_min, joint_max) in enumerate(self.joint_limits):
            # Check if proposed joint position violates limits
            proposed_pos = new_joints[i]
            
            if proposed_pos < joint_min:
                # Adjust action to stay within limits
                safe_action = max(raw_action[i], (joint_min - current_joints[i]) / 0.1)
            elif proposed_pos > joint_max:
                # Adjust action to stay within limits
                safe_action = min(raw_action[i], (joint_max - current_joints[i]) / 0.1)
            else:
                safe_action = raw_action[i]
            
            # Also ensure velocity limits are respected
            velocity = safe_action  # Our action directly represents desired change
            safe_action = np.clip(safe_action, self.velocity_limits[i][0], self.velocity_limits[i][1])
            
            safe_actions.append(safe_action)
        
        return np.array(safe_actions)

# Example usage
if __name__ == "__main__":
    # Define joint limits for a 3-DOF manipulator
    joint_limits = [(-np.pi/2, np.pi/2), (-np.pi/3, np.pi/3), (-np.pi/4, np.pi/4)]
    velocity_limits = [(-0.5, 0.5), (-0.3, 0.3), (-0.2, 0.2)]
    
    safe_agent = ConstrainedDDPGAgent(
        state_size=6, 
        action_size=3, 
        joint_limits=joint_limits, 
        velocity_limits=velocity_limits
    )
    
    # Simulate with current joint configuration
    current_joints = np.array([0.1, -0.2, 0.3])
    state = np.concatenate([current_joints, np.zeros(3)])  # Include velocities
    safe_action = safe_agent.act_with_constraints(state, current_joints[:3])
    
    print(f"Current joints: {current_joints}")
    print(f"Safe action: {safe_action}")
```

## Implementing Reward Functions

Designing appropriate reward functions is critical for successful RL in robotics:

### Types of Reward Functions

1. **Distance-based**: Reward inversely proportional to distance from goal
2. **Task-based**: Reward for completing specific subtasks
3. **Shaped rewards**: Intermediate rewards to guide learning
4. **Sparse rewards**: Reward only at task completion

### Example: Multi-component Reward Function

```python
class RewardFunction:
    def __init__(self, weights=None):
        self.weights = weights or {
            'distance': -1.0,     # Negative distance to goal
            'smoothness': -0.1,   # Penalty for jerky movements
            'energy': -0.05,      # Penalty for high energy consumption
            'success': 10.0       # Bonus for completing task
        }
    
    def calculate(self, state, action, next_state, goal, success=False):
        """Calculate reward based on multiple factors"""
        components = {}
        
        # Distance component (negative distance to goal)
        distance_to_goal = np.linalg.norm(next_state[:3] - goal)
        components['distance'] = self.weights['distance'] * distance_to_goal
        
        # Smoothness component (penalize large action changes)
        if hasattr(self, 'prev_action'):
            action_change = np.linalg.norm(action - self.prev_action)
            components['smoothness'] = self.weights['smoothness'] * action_change
        else:
            components['smoothness'] = 0
        self.prev_action = action.copy()
        
        # Energy component (penalize large actions)
        energy_cost = np.sum(np.abs(action))
        components['energy'] = self.weights['energy'] * energy_cost
        
        # Success component
        components['success'] = self.weights['success'] if success else 0
        
        # Calculate total reward
        total_reward = sum(components.values())
        
        return total_reward, components

# Example usage
reward_func = RewardFunction()
state = np.array([0.1, 0.2, 0.1, 0.0, 0.0, 0.0])
action = np.array([0.2, -0.1, 0.05])
next_state = np.array([0.15, 0.18, 0.12, 0.01, -0.02, 0.03])
goal = np.array([0.5, 0.3, -0.2])

reward, components = reward_func.calculate(state, action, next_state, goal)
print(f"Total reward: {reward}")
print(f"Reward components: {components}")
```

## Summary

This chapter covered reinforcement learning methods specifically adapted for robotics, including policy gradient methods and deep RL approaches like DDPG. We demonstrated practical implementations with safety considerations and proper reward function design. The examples provide a foundation that can be extended to more complex robotic tasks.

## Exercises

1. Implement a Twin Delayed DDPG (TD3) algorithm for robotic control
2. Design a reward function for a mobile robot navigation task that includes obstacle avoidance
3. Create a model-based RL approach that learns the robot's dynamics for safer exploration

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., ... & Wierstra, D. (2015). Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971.
- Levine, S., Finn, C., Darrell, T., & Abbeel, P. (2016). End-to-end training of deep visuomotor policies. The Journal of Machine Learning Research, 17(1), 1334-1373.
- Rajeswaran, A., Kumar, V., Gupta, A., & Todorov, E. (2017). Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. arXiv preprint arXiv:1709.10087.