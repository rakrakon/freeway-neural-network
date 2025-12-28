from env.freeway_env import FreewayENV
from graphics import Graphics


def main():
    env = FreewayENV()
    graphics = Graphics()
    obs, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        observation, reward, terminated, truncated, _ = env.step()
        graphics.render(env)
        print(info)


if __name__ == "__main__":
    main()
