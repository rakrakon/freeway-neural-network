import random


class ActionSpace:
    """Simple discrete action space compatible with gym-like interface"""

    def __init__(self, n):
        self.n = n

    def sample(self):
        """Sample a random action"""
        return random.randint(0, self.n - 1)

    def seed(self, seed=None):
        """Seed the action space RNG"""
        random.seed(seed)