import pygame

from env.freeway_env import FreewayENV
from graphics import Graphics


def main():
    env = FreewayENV()
    graphics = Graphics()
    obs, info = env.reset()
    done = False

    action_map = {
        pygame.K_w: 1, pygame.K_UP: 1,
        pygame.K_s: 2, pygame.K_DOWN: 2,
    }

    action = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        if done:
            break

        keys = pygame.key.get_pressed()
        action = 0
        for key, act in action_map.items():
            if keys[key]:
                action = act
                break

        observation, reward, done, info = env.step(action)
        graphics.render(observation, info['score'])


if __name__ == "__main__":
    main()