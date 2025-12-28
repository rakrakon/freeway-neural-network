import torch
import torch.nn as nn
import torch.nn.functional as F

# https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
# https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

class DQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()


        # Atari-style convolutional network
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, num_actions)

    def forward(self, x):
        # x shape: (batch, 4, 84, 84)
        x = x / 255.0  # normalize

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        return self.head(x)
