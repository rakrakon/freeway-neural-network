import pygame

from env.freeway_env import FreewayENV
from graphics import Graphics


def main():
    env = FreewayENV()
    graphics = Graphics()
    obs, info = env.reset()
    terminated = False
    truncated = False

    action_map = {
        pygame.K_w: 1, pygame.K_UP: 1,
        pygame.K_s: 2, pygame.K_DOWN: 2,
    }

    action = 0

    while not (terminated or truncated):

        # Handle keyboard input for human play
        if pygame.get_init():
            keys = pygame.key.get_pressed()
            for key, act in action_map.items():
                if keys[key]:
                    action = act
                    break
            else:
                action = 0

        observation, reward, terminated, truncated, info = env.step(action)
        graphics.render(env)
        print(info)


if __name__ == "__main__":
    main()
