import logging
from collections import deque
from datetime import datetime

import numpy as np
import torch

from config.config import cfg
from neuralnet.dqn import DuelingDQN
from utils.preprocessing import preprocess_frame


class DQNPlayer:
    def __init__(self, action_space_n: int, model_path = "best_model_freeway.pth"):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.train_config = cfg.train

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

        self.policy_net = DuelingDQN(
            h=self.train_config.frame_height,
            w=self.train_config.frame_width,
            outputs=action_space_n,
            frame_stack=self.train_config.frame_stack
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.policy_net.eval()

        logging.info(f"Loaded model from episode {checkpoint['episode']}")
        logging.info(f"Model's average reward: {checkpoint['avg_reward']:.2f}")

        self.frame_buffer = deque(maxlen=self.train_config.frame_stack)

    # Call this before calling get action!
    def reset(self, initial_observation):
        """Reset frame buffer with initial observation - fill with copies of first frame"""
        self.frame_buffer.clear()
        processed = preprocess_frame(initial_observation, self.train_config)

        # Fill buffer with copies of the first frame
        for _ in range(self.train_config.frame_stack):
            self.frame_buffer.append(processed.copy())


    def get_action(self, observation):
        processed = preprocess_frame(observation, self.train_config)
        self.frame_buffer.append(processed)

        while len(self.frame_buffer) < self.train_config.frame_stack:
            self.frame_buffer.append(processed.copy())

        # Stack frames and convert to tensor
        state = np.stack(self.frame_buffer, axis=0)
        state_tensor = torch.tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)

        # Get greedy action
        with torch.no_grad():
            action = self.policy_net(state_tensor).max(1).indices.item()

        return action

