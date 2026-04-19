from dataclasses import dataclass

import numpy as np

from config.config import Config
from env.action_space import ActionSpace
from utils.config_utils import load_config

PIXELS_PER_STEP = 1
SPEED_SCALE = 0.25
CAR_SIZE = (6, 3)
PLAYER_SIZE = (2, 2)

@dataclass
class Car:
    x: float
    y: float
    lane: int
    speed: int
    direction: int # 1 or -1


class FreewayENV:
    def __init__(self):
        self.config: Config = load_config()

        self.height = self.config.env.height
        self.width: int = self.config.env.width

        self.action_space = ActionSpace(3) # UP, DOWN, NOOP

        # We want the player in the middle of screen horizontally
        self.player_x: int = self.width // 2
        # And in the middle of the bottom sidewalk (7 pixels in height) so 4 is the middle
        self.player_y: int = self.height - 4
        self.prev_player_y = self.height - 4
        self.score = 0

        self.cars = []

        self.frame = 0
        self.max_frames = 2000
        self.skip_rate = 2 # Move every 2 frames

        self.reset()

    def reset(self):
        self._reset_player()
        self.score = 0
        self.frame = 0

        self.cars.clear()
        self._init_cars()

        return self._get_obs(), {}

    def _init_cars(self):
        lane_speeds = [1, 2, 1, 2, 1, 3, 2, 1, 2, 1]

        spacing = self.config.env.width

        for i in range(self.config.env.num_lanes):
            y = 11 + 7 * i

            speed = lane_speeds[i]
            direction = 1 if i % 2 == 0 else -1

            x = (i * (spacing // self.config.env.num_lanes)) % spacing

            self.cars.append(
                Car(
                    x=x,
                    y=y,
                    lane=i,
                    speed=speed,
                    direction=direction
                )
            )

    def _reset_player(self):
        self.prev_player_y = self.height - 4
        self.player_y = self.height - 4

    def _move_cars(self):
        for car in self.cars:
            # update position
            car.x += car.speed * car.direction * SPEED_SCALE

            # wrap-around
            if car.direction == 1:
                if car.x > self.width:
                    car.x = -CAR_SIZE[0]
            else:
                if car.x < -CAR_SIZE[0]:
                    car.x = self.width

    def _draw_cars(self, frame):
        for car in self.cars:
            x = int(car.x)
            y = int(car.y)

            for dx in range(CAR_SIZE[0]):
                for dy in range(CAR_SIZE[1]):
                    px = x + dx
                    py = y + dy

                    if 0 <= px < self.width and 0 <= py < self.height:
                        frame[py, px, 0] = 255 # Red

    def _draw_player(self, frame):
        for dx in range(2):
            for dy in range(2):
                px = self.player_x + dx
                py = self.player_y + dy

                if 0 <= px < self.width and 0 <= py < self.height:
                    frame[py, px, 1] = 255 # Green

    def _get_obs(self):
        # RGB frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self._draw_cars(frame)
        self._draw_player(frame)

        return frame

    def step(self, action):
        self.frame += 1

        self.prev_player_y = self.player_y

        self._move_cars()

        if self.frame % self.skip_rate == 0:
            # Move player
            if action == 1:
                self.player_y -= PIXELS_PER_STEP
            elif action == 2:
                self.player_y += PIXELS_PER_STEP

        # Keep player inside screen
        self.player_y = max(0, min(self.height - 3, self.player_y))

        has_collided = self._check_collision()

        reward = 0
        done = False

        # progress shaping
        reward += (self.prev_player_y - self.player_y) * 0.01

        if has_collided:
            reward = -1
            self.score -= 1
            self._reset_player() # Reset player to initial position

        # Player has reached the top
        if self.player_y <= 3:
            reward = 1
            self.score += 1
            self._reset_player() # Reset player to initial position

        if self.frame >= self.max_frames:
            done = True

        obs = self._get_obs()

        return obs, reward, done, { "score": self.score }

    def _check_collision(self) -> bool:
        px1 = self.player_x
        py1 = self.player_y
        px2 = self.player_x + PLAYER_SIZE[0]
        py2 = self.player_y + PLAYER_SIZE[1]

        for car in self.cars:
            cx1 = int(car.x)
            cy1 = int(car.y)
            cx2 = cx1 + CAR_SIZE[0]
            cy2 = cy1 + CAR_SIZE[1]

            # overlap check
            if (
                    px1 < cx2 and
                    px2 > cx1 and
                    py1 < cy2 and
                    py2 > cy1
            ):
                return True

        return False