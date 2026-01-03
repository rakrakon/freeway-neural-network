from env.freeway_env import FreewayENV
from graphics import Graphics
from neuralnet.dqn_player import DQNPlayer


def main():
    env = FreewayENV()
    graphics = Graphics()
    nn_player = DQNPlayer(env.action_space.n, "best_model_freeway.pth")
    observation, info = env.reset()
    terminated = False
    truncated = False

    nn_player.reset(observation)

    while not (terminated or truncated):
        action = nn_player.get_action(observation)
        print(f"Action: {action}")
        observation, reward, terminated, truncated, info = env.step(action)
        graphics.render(env)
        print(info)
    print(f"\nFinished with result: {info}")

if __name__ == "__main__":
    main()