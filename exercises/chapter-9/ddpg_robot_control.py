"""
Exercise: DDPG Implementation for Robotic Control
Chapter 9: Reinforcement Learning in Robotics

This exercise guides you through implementing Deep Deterministic 
Policy Gradient (DDPG) for robotic control tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque


class Actor(nn.Module):
    """
    Actor network: learns the policy (action selection) in DDPG.
    Maps states to actions.
    """
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh()  # Output actions bounded between -1 and 1
        )
    
    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    """
    Critic network: learns the Q-value function in DDPG.
    Maps (state, action) pairs to Q-values.
    """
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.state_input = nn.Linear(state_size, hidden_size)
        self.action_input = nn.Linear(action_size, hidden_size)
        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Q-value output
        )
    
    def forward(self, state, action):
        state_out = self.state_input(state)
        action_out = self.action_input(action)
        combined = torch.cat([state_out, action_out], dim=1)
        return self.network(combined)


class ReplayBuffer:
    """
    Experience replay buffer to store and sample experiences.
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        """Get buffer size."""
        return len(self.buffer)


class OUNoise:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated noise.
    Used for exploration in DDPG continuous action spaces.
    """
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        """Reset the noise."""
        self.state = self.mu.copy()
    
    def sample(self):
        """Generate noise."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) agent for continuous control.
    """
    def __init__(self, state_size, action_size, lr_actor=1e-4, lr_critic=1e-3, 
                 gamma=0.99, tau=1e-3, buffer_size=100000, batch_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update parameter
        self.batch_size = batch_size
        
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
        self.memory = ReplayBuffer(capacity=buffer_size)
        
        # Noise process for exploration
        self.noise = OUNoise(action_size)
    
    def hard_update(self, target, source):
        """Hard update target network with source network weights."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source):
        """Soft update target network with source network weights."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def act(self, state, add_noise=True, noise_scale=1.0):
        """Select action based on current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action from actor network
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train()
        
        # Add exploration noise
        if add_noise:
            noise = self.noise.sample() * noise_scale
            action += noise
        
        # Clip to valid action bounds (assumed to be [-1, 1])
        return np.clip(action, -1, 1)
    
    def reset_noise(self):
        """Reset the noise process."""
        self.noise.reset()
    
    def step(self, state, action, reward, next_state, done):
        """Store experience and update networks."""
        # Add experience to replay buffer
        self.memory.push(state, action, reward, next_state, done)
        
        # Learn from experiences if enough samples are available
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
    
    def learn(self, experiences):
        """Update networks based on experiences from replay buffer."""
        states, actions, rewards, next_states, dones = experiences
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Compute target Q-values for next states using target actor and critic
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions.detach())
        
        # Compute target Q-values: R + γ * Q_target(s', μ_target(s'))
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Update critic: minimize (target_q_values - critic(states, actions))^2
        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip gradients to prevent large updates
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor: maximize Q(state, actor(state)) (gradient ascent)
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Clip gradients to prevent large updates
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)


class SimpleRoboticArmEnv:
    """
    Simplified environment for a 2-DOF robotic arm.
    The goal is to move the end-effector to a target position.
    """
    def __init__(self):
        self.arm_length = [0.5, 0.4]  # Length of each arm segment
        self.state_size = 4  # [joint1_angle, joint2_angle, target_x, target_y]
        self.action_size = 2  # [change_in_joint1, change_in_joint2]
        self.max_steps = 100
        self.step_count = 0
        
        # Reset environment
        self.reset()
    
    def reset(self):
        """Reset the environment to a random state."""
        # Random starting joint angles
        self.joint1 = np.random.uniform(-np.pi/2, np.pi/2)
        self.joint2 = np.random.uniform(-np.pi/2, np.pi/2)
        
        # Random target position (within reachable range)
        self.target_x = np.random.uniform(-0.8, 0.8)
        self.target_y = np.random.uniform(-0.2, 0.8)  # Mostly upper half-plane
        
        self.step_count = 0
        self.prev_dist = None
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation."""
        return np.array([self.joint1, self.joint2, self.target_x, self.target_y])
    
    def _get_end_effector_position(self):
        """Calculate end-effector position based on joint angles."""
        # Forward kinematics for 2-DOF arm
        x = self.arm_length[0] * np.cos(self.joint1) + \
            self.arm_length[1] * np.cos(self.joint1 + self.joint2)
        y = self.arm_length[0] * np.sin(self.joint1) + \
            self.arm_length[1] * np.sin(self.joint1 + self.joint2)
        return x, y
    
    def step(self, action):
        """
        Execute an action in the environment.
        
        Args:
            action: [delta_joint1, delta_joint2] (continuous values)
            
        Returns:
            next_state: New state after executing action
            reward: Scalar reward
            done: Whether the episode is finished
        """
        # Apply action (with scaling for smooth movement)
        self.joint1 += action[0] * 0.1  # Scale down action magnitude
        self.joint2 += action[1] * 0.1
        
        # Clamp joint angles to reasonable ranges
        self.joint1 = np.clip(self.joint1, -np.pi, np.pi)
        self.joint2 = np.clip(self.joint2, -np.pi, np.pi)
        
        # Get end-effector position
        ee_x, ee_y = self._get_end_effector_position()
        
        # Calculate distance to target
        dist_to_target = np.sqrt((ee_x - self.target_x)**2 + (ee_y - self.target_y)**2)
        
        # Reward is negative distance with shaping to encourage progress
        reward = -dist_to_target
        
        # Add shaping reward based on improvement toward target
        if self.prev_dist is not None:
            if dist_to_target < self.prev_dist:
                reward += 0.1  # Small bonus for improvement
            else:
                reward -= 0.05  # Small penalty for getting farther
        
        self.prev_dist = dist_to_target
        
        # Check if target is reached or max steps exceeded
        done = dist_to_target < 0.1 or self.step_count >= self.max_steps
        self.step_count += 1
        
        return self._get_state(), reward, done, {}


def train_ddpg_agent(episodes=500, max_timesteps=200):
    """
    Train the DDPG agent on the robotic arm environment.
    
    Args:
        episodes: Number of training episodes
        max_timesteps: Maximum timesteps per episode
        
    Returns:
        agent: Trained DDPG agent
        rewards: List of total rewards per episode
    """
    env = SimpleRoboticArmEnv()
    agent = DDPGAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=1e-3
    )
    
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        agent.reset_noise()  # Reset noise for new episode
        
        for t in range(max_timesteps):
            # Select action with noise for exploration
            action = agent.act(state, add_noise=True, noise_scale=max(0.2, 0.2 * (1 - episode/500)))
            
            # Execute action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store experience and learn
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        rewards.append(total_reward)
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            print(f"Episode {episode}, Average Reward (last 100): {avg_reward:.2f}")
    
    return agent, rewards


def test_ddpg_agent(agent, episodes=5, max_timesteps=150):
    """
    Test the trained DDPG agent.
    
    Args:
        agent: Trained DDPG agent
        episodes: Number of test episodes
        max_timesteps: Maximum timesteps per episode
    """
    env = SimpleRoboticArmEnv()
    
    success_count = 0
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        trajectory = []  # Store trajectory for visualization
        
        # Store initial position
        ee_x, ee_y = env._get_end_effector_position()
        trajectory.append((ee_x, ee_y))
        
        print(f"\nTest Episode {episode + 1}")
        print(f"Target: ({env.target_x:.2f}, {env.target_y:.2f})")
        print(f"Initial EE: ({ee_x:.2f}, {ee_y:.2f})")
        
        for t in range(max_timesteps):
            # Select action without noise for testing
            action = agent.act(state, add_noise=False)
            
            state, reward, done, _ = env.step(action)
            
            # Store trajectory position
            ee_x, ee_y = env._get_end_effector_position()
            trajectory.append((ee_x, ee_y))
            
            total_reward += reward
            
            if done:
                break
        
        final_dist = np.sqrt((ee_x - env.target_x)**2 + (ee_y - env.target_y)**2)
        success = final_dist < 0.1
        if success:
            success_count += 1
        
        print(f"  Final: EE at ({ee_x:.2f}, {ee_y:.2f})")
        print(f"  Final distance: {final_dist:.3f}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Success: {'Yes' if success else 'No'}")
        
        # Plot the trajectory
        traj_x, traj_y = zip(*trajectory)
        plt.figure(figsize=(8, 8))
        plt.plot(traj_x, traj_y, 'b-', linewidth=2, label='Robot Path')
        plt.plot(env.target_x, env.target_y, 'go', markersize=15, label='Target')
        plt.plot(trajectory[0][0], trajectory[0][1], 'ro', markersize=10, label='Start')
        plt.plot(trajectory[-1][0], trajectory[-1][1], 'mo', markersize=10, label='End')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title(f'Test Episode {episode + 1} - Success: {success}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.show()
    
    success_rate = success_count / episodes
    print(f"\nOverall Success Rate: {success_rate:.1%} ({success_count}/{episodes})")


def main():
    """Main function to run the DDPG implementation exercise."""
    print("Exercise: DDPG Implementation for Robotic Control")
    print("Chapter 9: Reinforcement Learning in Robotics")
    print("=" * 60)
    
    # Train the DDPG agent
    print("\nTraining DDPG Agent...")
    agent, rewards = train_ddpg_agent(episodes=300)
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Total rewards per episode
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Smoothed rewards
    if len(rewards) > 50:
        window_size = 50
        smoothed_rewards = [np.mean(rewards[i:i+window_size]) 
                           for i in range(len(rewards) - window_size + 1)]
        plt.subplot(1, 2, 2)
        plt.plot(smoothed_rewards)
        plt.title('Smoothed Total Reward (Window = 50)')
        plt.xlabel('Episode')
        plt.ylabel('Average Total Reward')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTraining completed!")
    print(f"Final average reward (last 50 episodes): {np.mean(rewards[-50:]):.2f}")
    
    # Test the trained agent
    print("\nTesting Trained DDPG Agent...")
    test_ddpg_agent(agent, episodes=3)
    
    print("\n" + "="*60)
    print("Exercise Summary:")
    print("- Implemented Actor and Critic networks for DDPG")
    print("- Created experience replay buffer and OUNoise for exploration")
    print("- Trained DDPG agent on robotic arm control task")
    print("- Evaluated performance and visualized results")
    print("="*60)


if __name__ == "__main__":
    main()