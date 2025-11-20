from env.freeway_env import FreewayENV
from graphics import Graphics


def main():
    env = FreewayENV()
    graphics = Graphics()
    obs, info = env.reset()
    done = False
    while not done:
        _, done, info = env.step()
        graphics.render(env)
        print(info)

if __name__ == "__main__":
    main()
