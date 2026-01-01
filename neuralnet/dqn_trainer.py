import math
import random
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from datetime import datetime
from env.freeway_env import FreewayENV
from neuralnet.dqn import DuelingDQN
from neuralnet.replay_memory import ReplayMemory, Transition
from config.config import cfg
import cv2


class DQNTrainer:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.env = FreewayENV()
        self.train_config = cfg.train

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()

        # if GPU is to be used
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        self.episode_durations = []
        self.episode_rewards = []

        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

        self.frame_buffer = deque(maxlen=self.train_config.frame_stack)
        self.steps_done = 0

        # Get number of actions from gym action space
        n_actions = self.env.action_space.n

        # Initialize DuelingDQN with proper dimensions for stacked frames
        # Input: (frame_stack, frame_height, frame_width)
        self.policy_net = DuelingDQN(
            h=self.train_config.frame_height,
            w=self.train_config.frame_width,
            outputs=n_actions,
            frame_stack=self.train_config.frame_stack
        ).to(self.device)

        self.target_net = DuelingDQN(
            h=self.train_config.frame_height,
            w=self.train_config.frame_width,
            outputs=n_actions,
            frame_stack=self.train_config.frame_stack
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.train_config.learning_rate,
            amsgrad=True
        )
        self.memory = ReplayMemory(self.train_config.replay_buffer_size)

    def preprocess_frame(self, frame):
        """Preprocess Atari frame: grayscale, resize, normalize"""
        # Convert to grayscale
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])

        # Resize to configured dimensions (84x84)
        resized = cv2.resize(
            gray,
            (self.train_config.frame_width, self.train_config.frame_height),
            interpolation=cv2.INTER_AREA
        )

        # Keep as uint8 to save memory (normalize in model forward)
        return resized.astype(np.uint8)

    def get_state(self, observation):
        """Convert observation to stacked frames tensor"""
        # Preprocess the current frame
        processed = self.preprocess_frame(observation)

        # Add to frame buffer
        self.frame_buffer.append(processed)

        # If starting episode, fill buffer with copies of first frame
        while len(self.frame_buffer) < self.train_config.frame_stack:
            self.frame_buffer.append(processed)

        # Stack frames: (frame_stack, H, W)
        state = np.stack(self.frame_buffer, axis=0)

        return torch.tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)

    def reset_frame_buffer(self):
        """Call this at the start of each episode"""
        self.frame_buffer.clear()

    def select_action(self, state):
        sample = random.random()

        eps_threshold = self.train_config.epsilon_end + \
                        (self.train_config.epsilon_start - self.train_config.epsilon_end) * \
                        math.exp(-1. * self.steps_done / self.train_config.epsilon_decay)

        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def get_epsilon(self):
        """Helper to track current epsilon (useful for logging)"""
        return self.train_config.epsilon_end + \
            (self.train_config.epsilon_start - self.train_config.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.train_config.epsilon_decay)

    def optimize_model(self):
        """Optimize model using Double DQN with proper Atari settings"""
        batch_size = self.train_config.batch_size

        if len(self.memory) < batch_size:
            return None

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Create mask for non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool
        )

        # Concatenate non-final next states
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Clip rewards to [-1, 1] for stability
        reward_batch = torch.clamp(reward_batch, -1.0, 1.0)

        # Compute Q(s_t, a) - the model's current predictions
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Double DQN: use policy net to SELECT actions, target net to EVALUATE them
        next_state_values = torch.zeros(batch_size, device=self.device)
        if non_final_next_states.size(0) > 0:
            with torch.no_grad():
                # Policy net selects best actions
                next_actions = self.policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
                # Target net evaluates those actions
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(
                    1, next_actions
                ).squeeze(1)

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.train_config.gamma) + reward_batch

        # Compute Huber loss (smooth L1 loss)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping with proper value for Atari
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)

        self.optimizer.step()

        return loss.item()

    def train(self):
        """Train DQN agent on Atari Freeway with proper hyperparameters"""

        num_episodes = 50000 if torch.cuda.is_available() or torch.backends.mps.is_available() else 1000

        # Training metrics
        episode_rewards = []
        episode_lengths = []
        best_avg_reward = -float('inf')
        losses = []

        terminated = False
        truncated = False

        global_step = 0

        for i_episode in range(num_episodes):
            self.reset_frame_buffer()

            # Initialize episode
            observation, info = self.env.reset()
            state = self.get_state(observation)

            episode_reward = 0
            episode_loss = []

            for t in count():
                # Select and perform action
                action = self.select_action(state)

                # Frame skipping
                total_reward = 0
                for _ in range(4):
                    observation, reward, terminated, truncated, _ = self.env.step(action.item())
                    total_reward += reward
                    if terminated or truncated:
                        break

                reward_tensor = torch.tensor([total_reward], device=self.device)
                episode_reward += total_reward
                done = terminated or truncated

                # Get next state
                if terminated:
                    next_state = None
                else:
                    next_state = self.get_state(observation)

                # Store transition
                self.memory.push(state, action, next_state, reward_tensor)
                state = next_state

                global_step += 1

                # Start training after collecting initial experience
                if global_step >= self.train_config.learning_starts:
                    # Train every 4 steps
                    if global_step % 4 == 0:
                        loss = self.optimize_model()
                        if loss is not None:
                            episode_loss.append(loss)

                    # Hard update target network
                    if global_step % self.train_config.target_update_frequency == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                        self.logger.info(f"Target network updated at step {global_step}")

                if done:
                    break

            # Episode finished - record metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(t + 1)
            self.episode_durations.append(t + 1)
            self.episode_rewards.append(episode_reward)
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            losses.append(avg_loss)

            # Log progress every 10 episodes
            if (i_episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(
                    episode_rewards)
                avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(
                    episode_lengths)
                current_epsilon = self.get_epsilon()

                self.logger.info(
                    f"Episode {i_episode + 1}/{num_episodes} | "
                    f"Avg Reward (100ep): {avg_reward:.2f} | "
                    f"Avg Length: {avg_length:.1f} | "
                    f"Epsilon: {current_epsilon:.3f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Steps: {global_step}"
                )

                # Save best model
                if avg_reward > best_avg_reward and len(episode_rewards) >= 100:
                    best_avg_reward = avg_reward
                    torch.save({
                        'episode': i_episode,
                        'policy_net_state_dict': self.policy_net.state_dict(),
                        'target_net_state_dict': self.target_net.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'avg_reward': avg_reward,
                        'global_step': global_step,
                    }, 'best_model_freeway.pth')
                    self.logger.info(f"New best model saved! Avg reward: {avg_reward:.2f}")

            # Plot progress every 50 episodes
            # if (i_episode + 1) % 50 == 0:
            #     self.plot_training_progress(episode_rewards, episode_lengths, losses)

        self.logger.info('Training Complete')
        self.plot_training_progress(episode_rewards, episode_lengths, losses, show_result=True)

        # Save final model
        torch.save({
            'episode': num_episodes,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': episode_rewards,
        }, 'final_model_freeway.pth')

    def plot_training_progress(self, rewards, lengths, losses, show_result=False):
        """Plot training metrics"""
        plt.figure(figsize=(15, 5))

        # Plot rewards
        plt.subplot(1, 3, 1)
        plt.plot(rewards, alpha=0.3, label='Episode Reward')
        if len(rewards) >= 100:
            moving_avg = np.convolve(rewards, np.ones(100) / 100, mode='valid')
            plt.plot(range(99, 99 + len(moving_avg)), moving_avg, label='100-ep avg', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot episode lengths
        plt.subplot(1, 3, 2)
        plt.plot(lengths, alpha=0.3, label='Episode Length')
        if len(lengths) >= 100:
            moving_avg = np.convolve(lengths, np.ones(100) / 100, mode='valid')
            plt.plot(range(99, 99 + len(moving_avg)), moving_avg, label='100-ep avg', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Lengths')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot losses
        plt.subplot(1, 3, 3)
        if losses:
            plt.plot(losses, label='Training Loss')
            if len(losses) >= 100:
                moving_avg = np.convolve(losses, np.ones(100) / 100, mode='valid')
                plt.plot(range(99, 99 + len(moving_avg)), moving_avg, label='100-ep avg', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        if show_result:
            plt.savefig('freeway_training_results.png', dpi=150)
            self.logger.info("Training plots saved to 'freeway_training_results.png'")
        plt.pause(0.001)

    def plot_durations(self, show_result=False):
        """Legacy plotting method for compatibility"""
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        if self.is_ipython:
            if not show_result:
                pass
            else:
                pass