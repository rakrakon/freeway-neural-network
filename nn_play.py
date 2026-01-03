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

    total_reward = 0

    while not (terminated or truncated):
        action = nn_player.get_action(observation)
        print(f"Action: {action}")
        observation, reward, terminated, truncated, info = env.step(action)
        graphics.render(env)
        print(info)

        total_reward += reward
    print(f"\nFinished with result: {info}\n Total_reward: {total_reward}")

if __name__ == "__main__":
    main()