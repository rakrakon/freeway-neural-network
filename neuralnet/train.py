import math
import random
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
from graphics import Graphics
from neuralnet.replay_memory import ReplayMemory, Transition

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

env = FreewayENV()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

logger.info(f"Using device: {device}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")


# CNN-based DQN for image inputs
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32

        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.float() / 255.0  # Normalize pixel values
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
NUM_EPISODES = 600
TARGET_UPDATE = 10
OPTIMIZE_EVERY = 4

# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the dimensions of the observation
state, info = env.reset()
screen_height, screen_width, channels = state.shape

logger.info(f"Environment initialized")
logger.info(f"Screen dimensions: {screen_height}x{screen_width}x{channels}")
logger.info(f"Number of actions: {n_actions}")

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

logger.info(f"Networks initialized")
logger.info(f"Policy network parameters: {sum(p.numel() for p in policy_net.parameters())}")

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

logger.info("=" * 60)
logger.info("Training Configuration:")
logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
logger.info(f"  GAMMA: {GAMMA}")
logger.info(f"  EPS_START: {EPS_START}, EPS_END: {EPS_END}, EPS_DECAY: {EPS_DECAY}")
logger.info(f"  TAU: {TAU}")
logger.info(f"  Learning Rate: {LR}")
logger.info(f"  NUM_EPISODES: {NUM_EPISODES}")
logger.info(f"  Memory Size: 10000")
logger.info(f"  TARGET_UPDATE: {TARGET_UPDATE} episodes")
logger.info(f"  OPTIMIZE_EVERY: {OPTIMIZE_EVERY} steps")
logger.info("=" * 60)

steps_done = 0

graphics = Graphics

def get_state(observation):
    """Convert observation to tensor and transpose to (C, H, W) format"""
    # observation is (H, W, C), need to convert to (C, H, W) for PyTorch
    state = np.transpose(observation, (2, 0, 1))
    return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []
episode_rewards = []


def plot_durations(show_result=False, force_update=False):
    if not show_result and not force_update and len(episode_durations) % 10 != 0:
        return

    plt.figure(1, figsize=(12, 5))

    # Plot durations
    plt.subplot(1, 2, 1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result - Duration')
    else:
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.title('Training - Duration')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # Plot rewards
    plt.subplot(1, 2, 2)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result - Reward')
    else:
        plt.title('Training - Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(rewards_t.numpy())
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.tight_layout()
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None

    transitions = memory.sample(BATCH_SIZE)
    # Use list comprehension instead of zip for better performance
    batch = Transition(*zip(*transitions))

    # Build mask and collect non-final next states
    non_final_next_states = []
    non_final_indices = []
    for idx, s in enumerate(batch.next_state):
        if s is not None:
            non_final_next_states.append(s)
            non_final_indices.append(idx)

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - this is the model's prediction
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if non_final_next_states:
        non_final_next_states_batch = torch.cat(non_final_next_states)
        with torch.no_grad():
            next_values = target_net(non_final_next_states_batch).max(1).values
            for i, idx in enumerate(non_final_indices):
                next_state_values[idx] = next_values[i]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def calculate_reward(info, action, current_y, previous_y):
    # Big terminal reward
    if info.get("score", 0) > 0:
        return 100.0

    if info.get("collision", False):
        return -2.0

    if previous_y is None:
        return 0.0

    # Forward progress (positive if moving up)
    y_progress = previous_y - current_y

    # Reward proportional to movement
    return 0.5 * y_progress


# Training loop
logger.info("Starting training...")
best_score = 0
total_steps = 0

# TODO: Add frame skipping

for i_episode in range(NUM_EPISODES):
    observation, info = env.reset()
    state = get_state(observation)

    total_reward = 0
    episode_loss = 0
    loss_count = 0

    for t in range(5000):
        previous_y = env.chicken_center_y
        action = select_action(state)
        observation, done, info = env.step(action.item())
        total_steps += 1

        logger.info(f"Making step {t}..")

        reward = calculate_reward(info, action, env.chicken_center_y, previous_y)

        total_reward += reward
        reward = torch.tensor([reward], device=device)

        if done:
            next_state = None
        else:
            next_state = get_state(observation)

        memory.push(state, action, next_state, reward)
        state = next_state

        if total_steps % OPTIMIZE_EVERY == 0:
            logger.info(f"Optimizing..")
            loss = optimize_model()
            if loss is not None:
                episode_loss += loss
                loss_count += 1
            logger.info("Done.")

        if done:
            episode_durations.append(t + 1)
            episode_rewards.append(total_reward)
            plot_durations()
            break

    # Update target network less frequently for better performance
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Calculate average loss for the episode
    avg_loss = episode_loss / loss_count if loss_count > 0 else 0
    epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

    # Track best score
    if info['score'] > best_score:
        best_score = info['score']
        logger.info(f"New best score: {best_score}")
        torch.save(policy_net.state_dict(), 'best_policy_net.pth')

    if i_episode % 10 == 0:
        logger.info(
            f"Episode {i_episode}/{NUM_EPISODES} | "
            f"Duration: {t + 1} | "
            f"Reward: {total_reward:.2f} | "
            f"Score: {info['score']} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Total Steps: {total_steps}"
        )

    # Save checkpoint every 100 episodes
    if i_episode > 0 and i_episode % 100 == 0:
        checkpoint_path = f'checkpoint_episode_{i_episode}.pth'
        torch.save({
            'episode': i_episode,
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode_durations': episode_durations,
            'episode_rewards': episode_rewards,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

logger.info('Training Complete!')
logger.info(f"Best score achieved: {best_score}")
logger.info(f"Total training steps: {total_steps}")
plot_durations(show_result=True)
plt.ioff()
plt.show()

# Save the trained model
torch.save(policy_net.state_dict(), 'policy_net.pth')
logger.info("Final model saved as 'policy_net.pth'")
logger.info("Best model saved as 'best_policy_net.pth'")
logger.info(f"Training log saved to training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
