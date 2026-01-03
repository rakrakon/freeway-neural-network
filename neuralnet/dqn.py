import torch
import torch.nn as nn
import torch.nn.functional as F

# https://hyper.ai/en/sota/tasks/atari-games/benchmark/atari-games-on-atari-2600-freeway
# QR DQN - 34
# Bootstrapped DQN - 33.9
# Double DuelingDQN - 33.3

class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture - often performs better for Atari games.
    Separates state value and action advantages.
    """

    def __init__(self, h=84, w=84, outputs=3, frame_stack=4):
        super(DuelingDQN, self).__init__()

        # Shared convolutional layers
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        # Value stream
        self.value_fc1 = nn.Linear(linear_input_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Advantage stream
        self.advantage_fc1 = nn.Linear(linear_input_size, 256)
        self.advantage_fc2 = nn.Linear(256, outputs)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Shared convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # Value stream
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)

        # Advantage stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)

        # Combine value and advantage using the aggregation formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values