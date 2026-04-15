import pygame
import numpy as np

from env.freeway_env import FreewayENV
from neuralnet.dqn_player import DQNPlayer


class DebugViewer:
    def __init__(self, width=84, height=84, scale=6):
        pygame.init()
        self.scale = scale
        self.screen = pygame.display.set_mode(
            (width * scale, height * scale)
        )
        self.clock = pygame.time.Clock()

    def render(self, obs):
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))

        surf = pygame.transform.scale(
            surf,
            (surf.get_width() * self.scale, surf.get_height() * self.scale)
        )

        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(60)

    def process_events(self):
        action = 0  # NOOP by default

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        keys = pygame.key.get_pressed()

        # Forward (Up / W)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action = 1

        # Backward (Down / S)
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = 2

        return action

def main():
    env = FreewayENV()
    viewer = DebugViewer()
    obs, _ = env.reset()

    nn_player = DQNPlayer(env.action_space.n, "best_model_freeway.pth")

    total_reward = 0

    while True:
        # action = viewer.process_events()
        action = nn_player.get_action(obs)

        obs, reward, done, info = env.step(action)
        total_reward += reward

        viewer.render(obs)

        if done:
            print("Times up")
            print(total_reward)
            break

if __name__ == "__main__":
    main()