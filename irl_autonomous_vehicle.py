# Inverse Reinforcement Learning for Autonomous Vehicles
# A comprehensive implementation using Highway-Env

import numpy as np
import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from collections import deque, namedtuple
import os
import time  # Add this import

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Define hyperparameters
GAMMA = 0.99  # Discount factor
BATCH_SIZE = 64  # Batch size for training
LEARNING_RATE = 0.001  # Learning rate
BUFFER_SIZE = 100000  # Replay buffer size
TAU = 0.005  # Soft update parameter
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.1  # Final exploration rate
EPSILON_DECAY = 10000  # Decay steps
FEATURE_DIM = 5  # Dimensionality of feature vector
HIDDEN_DIM = 256  # Hidden layer size
EXPERT_DEMOS_FILE = os.path.join('data', 'expert_demos.npz')  # Expert demonstrations file
MODEL_DIR = 'models'
DATA_DIR = 'data'
VISUALIZATION_DIR = 'visualizations'

# Define the environment configuration
def configure_env():
    """Configure the Highway-Env environment"""
    config = {
        'observation': {
            'type': 'Kinematics',
            'vehicles_count': 5,
            'features': ['presence', 'x', 'y', 'vx', 'vy'],
            'normalize': True
        },
        'action': {
            'type': 'DiscreteMetaAction',
        },
        'lanes_count': 3,
        'initial_lane_id': 1,
        'vehicles_density': 1,
        'duration': 40,  # Episode length in seconds
        'simulation_frequency': 15,
        'policy_frequency': 5,
        'render_mode': 'rgb_array',
        'screen_width': 600,
        'screen_height': 150,
        'centering_position': [0.3, 0.5],
        'scaling': 5.5,
        'show_trajectories': True
    }

    env = gym.make('highway-v0', render_mode='rgb_array', config=config)
    env.reset()
    return env

# Feature extraction function
def extract_features(state):
    """
    Extract relevant features from the state for IRL
    These features capture driving behavior characteristics
    """
    # Initialize feature vector
    features = np.zeros(FEATURE_DIM)
    
    try:
        # State is now directly the observation array
        observation = state  # No need to get state[0]
        
        # Extract ego vehicle state (first vehicle)
        ego_vehicle = observation[0]  # First vehicle is the ego vehicle
        
        # Lane position (y-coordinate normalized between -1 and 1)
        features[0] = ego_vehicle[2]  # y position
        
        # Process other vehicles
        min_front_dist = float('inf')
        min_rear_dist = float('inf')
        front_speed_diff = 0
        
        # Start from index 1 to skip ego vehicle
        for vehicle in observation[1:]:
            if vehicle[0] > 0:  # Vehicle is present
                x_diff = vehicle[1] - ego_vehicle[1]  # Relative x position
                if x_diff > 0:  # Vehicle is ahead
                    if x_diff < min_front_dist:
                        min_front_dist = x_diff
                        front_speed_diff = vehicle[3] - ego_vehicle[3]  # vx difference
                else:  # Vehicle is behind
                    rear_dist = -x_diff
                    if rear_dist < min_rear_dist:
                        min_rear_dist = rear_dist
        
        # Assign features
        features[1] = float(min_front_dist if min_front_dist != float('inf') else 10.0)
        features[2] = float(front_speed_diff)
        features[3] = float(min_rear_dist if min_rear_dist != float('inf') else 10.0)
        features[4] = float(abs(ego_vehicle[4]))  # Use lateral velocity as proxy for lane changing
        
        return features
        
    except Exception as e:
        print("Error in feature extraction:")
        print(f"State: {state}")
        print(f"Exception: {str(e)}")
        raise e

# Neural Network for Q-function approximation
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Neural Network for reward function approximation
class RewardNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim=HIDDEN_DIM):
        super(RewardNetwork, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, features):
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "features", "next_features"])
        
    def add(self, state, action, reward, next_state, done, features, next_features):
        e = self.experience(state, action, reward, next_state, done, features, next_features)
        self.memory.append(e)
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        features = torch.from_numpy(np.vstack([e.features for e in experiences if e is not None])).float()
        next_features = torch.from_numpy(np.vstack([e.next_features for e in experiences if e is not None])).float()
        
        return states, actions, rewards, next_states, dones, features, next_features
    
    def __len__(self):
        return len(self.memory)

# Maximum Entropy IRL 
class MaxEntIRL:
    def __init__(self, feature_dim, learning_rate=0.001):
        self.reward_network = RewardNetwork(feature_dim)
        self.optimizer = optim.Adam(self.reward_network.parameters(), lr=learning_rate)
        
    def train_step(self, expert_features, policy_features):
        """
        Train the reward network for one step using the MaxEnt IRL objective
        """
        # Convert to torch tensors if they aren't already
        if not isinstance(expert_features, torch.Tensor):
            expert_features = torch.FloatTensor(expert_features)
        if not isinstance(policy_features, torch.Tensor):
            policy_features = torch.FloatTensor(policy_features)
            
        # Forward pass for expert and policy features
        expert_rewards = self.reward_network(expert_features)
        policy_rewards = self.reward_network(policy_features)
        
        # Maximum entropy objective: maximize reward on expert data, minimize on policy data
        loss = -torch.mean(expert_rewards) + torch.mean(policy_rewards)
        
        # Add L2 regularization to prevent reward magnitude explosion
        l2_reg = 0.001
        for param in self.reward_network.parameters():
            loss += l2_reg * torch.sum(param ** 2)
            
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_reward(self, features):
        """Predict reward for a given feature vector"""
        with torch.no_grad():
            if not isinstance(features, torch.Tensor):
                features = torch.FloatTensor(features)
            return self.reward_network(features).item()

# Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.qnetwork_local = QNetwork(state_dim, action_dim)
        self.qnetwork_target = QNetwork(state_dim, action_dim)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.t_step = 0
        self.epsilon = EPSILON_START
        self.action_dim = action_dim
        
    def step(self, state, action, reward, next_state, done, features, next_features):
        """Save experience to replay buffer and learn if enough samples are available"""
        # Save experience to replay buffer
        self.memory.add(state, action, reward, next_state, done, features, next_features)
        
        # Learn from experiences if enough samples are available
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
            
        # Update epsilon for exploration
        self.t_step += 1
        self.epsilon = max(EPSILON_END, EPSILON_START - (self.t_step / EPSILON_DECAY))
    
    def act(self, state, eval_mode=False):
        """Choose an action using epsilon-greedy policy"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon or eval_mode:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))
    
    def learn(self, experiences, gamma):
        """Update value parameters using batch of experience tuples"""
        states, actions, rewards, next_states, dones, _, _ = experiences
        
        # Get max predicted Q values for next states from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save(self, filename):
        """Save trained model"""
        torch.save(self.qnetwork_local.state_dict(), filename)
    
    def load(self, filename):
        """Load trained model"""
        self.qnetwork_local.load_state_dict(torch.load(filename))
        self.qnetwork_target.load_state_dict(torch.load(filename))

# Generate expert demonstrations (simulated for this example)
def generate_expert_demos(env, num_episodes=10):
    """
    Generate expert demonstrations using a rule-based controller
    In a real application, these would come from human demonstrations
    """
    demos = []
    
    for i in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle new Gymnasium reset() that returns (obs, info)
            state = state[0]
        done = False
        episode_states = []
        episode_actions = []
        episode_features = []
        
        while not done:
            # Expert policy (rule-based for simplicity)
            features = extract_features(state)
            
            # State is directly the observation array
            ego_vehicle = state[0]  # First row contains ego vehicle data
            ego_y = ego_vehicle[2]  # Lane position
            ego_speed = ego_vehicle[3]  # Longitudinal speed
            
            # Simple rule-based policy:
            # - If there's a car too close ahead, change lanes or slow down
            # - Otherwise, maintain speed in the center lane
            
            # Mapping: 0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER
            if features[1] < 0.5:  # Car close ahead
                if ego_y > 0:  # In right lane
                    action = 0  # Move left
                elif ego_y < 0:  # In left lane
                    action = 2  # Move right
                else:
                    action = 4  # Slow down
            elif abs(ego_y) > 0.2:  # Not in center lane
                if ego_y > 0:  # In right lane
                    action = 0  # Move left
                else:  # In left lane
                    action = 2  # Move right
            else:
                action = 3 if ego_speed < 0.8 else 1  # Speed up if slow, otherwise maintain
                
            # Handle new Gymnasium step() that returns (obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated  # Episode ends if either is True
            next_features = extract_features(next_state)
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_features.append(features)
            
            state = next_state
            
        demos.append((episode_states, episode_actions, episode_features))
        print(f"Generated expert demo {i+1}/{num_episodes}")
    
    return demos

# Save and load demonstrations
def save_demos(demos, filename):
    """
    Save demonstrations using numpy's savez format to handle variable-length episodes
    """
    # Convert demos into a dictionary of arrays
    demo_dict = {
        f'episode_{i}': {
            'states': np.array(demo[0]),
            'actions': np.array(demo[1]),
            'features': np.array(demo[2])
        }
        for i, demo in enumerate(demos)
    }
    
    # Save using numpy's savez
    np.savez_compressed(filename, **demo_dict)
    print(f"Saved {len(demos)} demonstrations to {filename}")

def load_demos(filename):
    """
    Load demonstrations from npz file and convert back to list format
    """
    data = np.load(filename, allow_pickle=True)
    demos = []
    
    # Sort keys to maintain order
    episode_keys = sorted([k for k in data.keys() if k.startswith('episode_')])
    
    for key in episode_keys:
        episode = data[key].item()  # Convert from numpy object array to dict
        demos.append((
            episode['states'],
            episode['actions'],
            episode['features']
        ))
    
    print(f"Loaded {len(demos)} demonstrations from {filename}")
    return demos

# Training loop for IRL
def train_irl(env, expert_demos, feature_dim, num_iterations=50):
    print("\nInitializing IRL training...")
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    
    irl = MaxEntIRL(feature_dim)
    agent = DQNAgent(state_dim, action_dim)
    
    print("Extracting expert features...")
    expert_features = []
    for demo in expert_demos:
        expert_features.extend(demo[2])
    expert_features = np.array(expert_features)
    
    losses = []
    rewards = []
    
    # Create a figure for live plotting
    plt.ion()  # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    loss_line, = ax1.plot([], [], 'b-', label='Loss')
    reward_line, = ax2.plot([], [], 'r-', label='Reward')
    
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax2.set_title('Average Reward')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Reward')
    
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    
    # Main training loop with progress bar
    pbar = tqdm(range(num_iterations), desc="Training IRL")
    for iteration in pbar:
        policy_features = []
        episode_rewards = []
        
        # Collect rollouts
        for episode in range(5):
            state, _ = env.reset()  # Unpack the tuple
            done = False
            episode_reward = 0
            
            while not done:
                flat_state = state.flatten()
                action = agent.act(flat_state)
                features = extract_features(state)
                
                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_features = extract_features(next_state)
                
                with torch.no_grad():
                    reward = irl.predict_reward(features)
                
                policy_features.append(features)
                episode_reward += reward
                
                agent.step(flat_state, action, reward, next_state.flatten(), done, features, next_features)
                state = next_state
            
            episode_rewards.append(episode_reward)
        
        # Update reward function
        policy_features = np.array(policy_features)
        batch_size = min(len(expert_features), len(policy_features))
        
        total_loss = 0
        num_batches = 10
        for _ in range(num_batches):
            expert_idx = np.random.choice(len(expert_features), batch_size)
            policy_idx = np.random.choice(len(policy_features), batch_size)
            loss = irl.train_step(expert_features[expert_idx], policy_features[policy_idx])
            total_loss += loss
            losses.append(loss)
        
        avg_loss = total_loss / num_batches
        avg_reward = np.mean(episode_rewards)
        rewards.append(avg_reward)
        
        # Update progress bar description
        pbar.set_description(f"Loss: {avg_loss:.4f}, Reward: {avg_reward:.2f}")
        
        # Update live plots
        loss_line.set_data(range(len(losses)), losses)
        reward_line.set_data(range(len(rewards)), rewards)
        
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        
        # Show current driving behavior every 5 iterations
        if iteration % 5 == 0:
            print("\nDemonstrating current driving behavior...")
            state, _ = env.reset()  # Unpack the tuple
            done = False
            env.render()  # Show the environment
            while not done:
                flat_state = state.flatten()
                action = agent.act(flat_state, eval_mode=True)
                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                env.render()
                time.sleep(0.1)  # Slow down visualization
            
            print("Demonstration complete. Continuing training...\n")
        
        plt.pause(0.01)  # Update the plots
        
        # Save checkpoints
        if iteration % 10 == 0:
            print(f"\nSaving checkpoint at iteration {iteration}")
            torch.save({
                'iteration': iteration,
                'agent_qnetwork_local_state_dict': agent.qnetwork_local.state_dict(),
                'agent_qnetwork_target_state_dict': agent.qnetwork_target.state_dict(),
                'irl_reward_network_state_dict': irl.reward_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'losses': losses,
                'rewards': rewards
            }, os.path.join(MODEL_DIR, f'checkpoint_iter_{iteration}.pth'))
    
    plt.ioff()  # Disable interactive mode
    return agent, irl, losses, rewards

# Evaluation function
def evaluate_policy(env, agent, irl, num_episodes=10, render=False):
    total_rewards = []
    success_rate = 0
    
    for i in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            if render:
                env.render()
            
            flat_state = state.flatten()
            action = agent.act(flat_state, eval_mode=True)
            features = extract_features(state)
            
            next_state, _, done, info = env.step(action)
            reward = irl.predict_reward(features)
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Check if episode was successful (reached goal without crashing)
            if done and steps >= env.config["duration"] * env.config["policy_frequency"]:
                success_rate += 1
        
        total_rewards.append(episode_reward)
        print(f"Episode {i+1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    success_rate = success_rate / num_episodes
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Success Rate: {success_rate:.2f}")
    
    return np.mean(total_rewards), success_rate

# Visualization functions
def visualize_training(losses, rewards):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('IRL Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig('irl_training_curves.png')
    plt.show()

def visualize_reward_function(irl, env):
    """Visualize the learned reward function across feature space"""
    # Create a grid of feature values
    num_points = 20
    
    # For simplicity, we'll visualize the reward as a function of 
    # distance to the vehicle ahead and lane position
    front_dist = np.linspace(0.1, 2.0, num_points)
    lane_pos = np.linspace(-1.0, 1.0, num_points)
    
    rewards = np.zeros((num_points, num_points))
    
    for i, fd in enumerate(front_dist):
        for j, lp in enumerate(lane_pos):
            # Create a feature vector
            features = np.zeros(FEATURE_DIM)
            features[0] = lp  # Lane position
            features[1] = fd  # Front distance
            features[2] = 0.0  # Speed difference
            features[3] = 2.0  # Rear distance (fixed)
            features[4] = 0.0  # Lane change penalty
            
            # Get reward prediction
            rewards[i, j] = irl.predict_reward(features)
    
    # Plot the reward landscape
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(lane_pos, front_dist)
    plt.pcolormesh(X, Y, rewards)
    plt.colorbar(label='Reward')
    plt.xlabel('Lane Position')
    plt.ylabel('Distance to Front Vehicle')
    plt.title('Learned Reward Function')
    plt.savefig('reward_function_visualization.png')
    plt.show()

def compare_policies(env, agent, irl, num_episodes=5):
    """Compare learned policy with rule-based and random policies"""
    # Evaluate learned policy
    irl_rewards, irl_success = evaluate_policy(env, agent, irl, num_episodes)
    
    # Evaluate rule-based policy
    rule_rewards = []
    rule_success = 0
    
    for i in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            features = extract_features(state)
            
            # Simple rule-based policy
            if features[1] < 0.5:  # Car close ahead
                if state[0][2] > 0:  # In right lane
                    action = 0  # Move left
                elif state[0][2] < 0:  # In left lane
                    action = 2  # Move right
                else:
                    action = 4  # Slow down
            elif abs(state[0][2]) > 0.2:  # Not in center lane
                if state[0][2] > 0:  # In right lane
                    action = 0  # Move left
                else:  # In left lane
                    action = 2  # Move right
            else:
                action = 3 if state[0][3] < 0.8 else 1  # Speed up if slow, otherwise maintain
            
            next_state, _, done, _ = env.step(action)
            reward = irl.predict_reward(features)
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done and steps >= env.config["duration"] * env.config["policy_frequency"]:
                rule_success += 1
        
        rule_rewards.append(episode_reward)
    
    rule_success = rule_success / num_episodes
    rule_rewards = np.mean(rule_rewards)
    
    # Evaluate random policy
    random_rewards = []
    random_success = 0
    
    for i in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            features = extract_features(state)
            action = env.action_space.sample()  # Random action
            
            next_state, _, done, _ = env.step(action)
            reward = irl.predict_reward(features)
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done and steps >= env.config["duration"] * env.config["policy_frequency"]:
                random_success += 1
        
        random_rewards.append(episode_reward)
    
    random_success = random_success / num_episodes
    random_rewards = np.mean(random_rewards)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    methods = ['IRL Policy', 'Rule-based Policy', 'Random Policy']
    reward_values = [irl_rewards, rule_rewards, random_rewards]
    success_values = [irl_success, rule_success, random_success]
    
    plt.subplot(1, 2, 1)
    plt.bar(methods, reward_values)
    plt.title('Average Reward')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.bar(methods, success_values)
    plt.title('Success Rate')
    plt.ylabel('Success Rate')
    
    plt.tight_layout()
    plt.savefig('policy_comparison.png')
    plt.show()

# Main function to run the project
def main():
    print("Setting up environment...")
    env = configure_env()
    
    # Create directories
    for directory in [DATA_DIR, MODEL_DIR, VISUALIZATION_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    expert_demos_path = os.path.join(DATA_DIR, 'expert_demos.npz')
    
    # Load or generate demonstrations
    try:
        if os.path.exists(expert_demos_path) and os.path.getsize(expert_demos_path) > 0:
            print("Loading existing expert demonstrations...")
            expert_demos = load_demos(expert_demos_path)
        else:
            print("Generating new expert demonstrations...")
            expert_demos = generate_expert_demos(env)
            save_demos(expert_demos, expert_demos_path)
    except (EOFError, ValueError) as e:
        print(f"Error with demonstrations file: {e}")
        print("Generating new expert demonstrations...")
        expert_demos = generate_expert_demos(env)
        save_demos(expert_demos, expert_demos_path)
    
    # Train IRL
    agent, irl, losses, rewards = train_irl(env, expert_demos, FEATURE_DIM, num_iterations=30)
    
    print("Saving final models...")
    torch.save({
        'agent_qnetwork_local_state_dict': agent.qnetwork_local.state_dict(),
        'agent_qnetwork_target_state_dict': agent.qnetwork_target.state_dict(),
        'irl_reward_network_state_dict': irl.reward_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, os.path.join(MODEL_DIR, 'final_model.pth'))
    
    print("Generating visualizations...")
    visualize_training(losses, rewards)
    visualize_reward_function(irl, env)
    
    print("Evaluating learned policy...")
    evaluate_policy(env, agent, irl, num_episodes=10, render=True)
    
    print("Comparing with baselines...")
    compare_policies(env, agent, irl)
    
    env.close()
    print("Done!")

if __name__ == "__main__":
    main()
