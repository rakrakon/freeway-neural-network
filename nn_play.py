import pygame

from env.freeway_env import FreewayENV
from graphics import Graphics
from neuralnet.dqn_player import DQNPlayer


def main():
    env = FreewayENV()
    graphics = Graphics()
    nn_player = DQNPlayer(env.action_space.n, "best_model_freeway.pth")

    observation, info = env.reset()
    done = False

    nn_player.reset(observation)

    total_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        if done:
            break

        action = nn_player.get_action(observation)
        print(f"Action: {action}")
        observation, reward, done, info = env.step(action)
        graphics.render(observation, info["score"])

        total_reward += reward

    print(f"Total_reward: {total_reward}")


if __name__ == "__main__":
    main()