from env.freeway_env import FreewayENV
from graphics import Graphics

graphics = Graphics()
env = FreewayENV()

for i in range(300):
    env.step(1)

while True:
    graphics.render(env)