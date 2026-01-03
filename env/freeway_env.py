import random
import numpy as np
import pygame

from env.action_space import ActionSpace
from utils.config_utils import load_config

ROAD_COLOR = [118, 122, 122]
CHICKEN_COLOR = [110, 84, 35]
CAR_COLOR = [0, 0, 255]
WHITE_COLOR = [255, 255, 255]
YELLOW_COLOR = [255, 255, 0]
SIDEWALK_COLOR = [180, 180, 180]
SIDEWALK_HEIGHT = 50  # Increased from 20 to give chicken safe starting zone

DIVIDER_THICKNESS = 2
SEPARATION = 4
DASH_LENGTH = 10
GAP_LENGTH = 10

class FreewayENV:
    def __init__(self):
        self.config = load_config()

        self.cars = []
        self.steps = 0
        self.score = 0
        self.done = False
        self.clock = None

        # Config
        self.screen_width = self.config['env']['width']
        self.screen_height = self.config['env']['height']
        self.num_lanes = self.config['env']['num_lanes']
        self.lane_height = self.screen_height // self.num_lanes
        self.fps = self.config['env']['fps']

        self.chicken_center_x = None
        self.chicken_center_y = None

        # MUCH LARGER sprites so they're visible after downscaling to 84x84
        self.chicken_width = 30
        self.chicken_height = 30

        self.car_width = 35
        self.car_height = 25

        # Collision boxes slightly smaller for fair gameplay
        self.chicken_collision_width = 25
        self.chicken_collision_height = 25

        self.reset()

        self.action_space = ActionSpace(3)

    def reset(self):
        self.cars = []
        self.steps = 0
        self.score = 0

        lane_height = (self.screen_height - 2 * SIDEWALK_HEIGHT) // self.num_lanes

        # Define safe spawn zone (avoid chicken's initial position)
        chicken_spawn_x = self.screen_width // 2
        safe_distance = 50  # pixels

        for i in range(self.num_lanes):
            lane_center = SIDEWALK_HEIGHT + i * lane_height + lane_height // 2

            # Only spawn cars far from chicken on the bottom lane (closest to starting position)
            is_bottom_lane = (i == self.num_lanes - 1)  # Last lane is closest to chicken start

            if is_bottom_lane:
                # Spawn far from chicken horizontally to avoid immediate collision
                if random.random() < 0.5:
                    car_x = random.uniform(0, chicken_spawn_x - safe_distance)
                else:
                    car_x = random.uniform(chicken_spawn_x + safe_distance, self.screen_width)
            else:
                car_x = random.uniform(0, self.screen_width)

            self.cars.append({
                'center_x': car_x,
                'center_y': lane_center,
                'speed': random.uniform(1.5, 3.0)
            })

        self.chicken_center_x = self.screen_width // 2
        self.chicken_center_y = self.screen_height - 25  # Centered in bottom sidewalk (50/2 = 25)

        return self.get_obs(), {'score': self.score}

    def get_obs(self):
        obs = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        self.draw_lanes(obs)

        cx, cy = int(self.chicken_center_x), int(self.chicken_center_y)
        obs[cy - self.chicken_height // 2: cy + self.chicken_height // 2 + 1,
        cx - self.chicken_width // 2: cx + self.chicken_width // 2 + 1] = CHICKEN_COLOR

        for car in self.cars:
            cx, cy = int(car['center_x']), int(car['center_y'])
            obs[cy - self.car_height // 2: cy + self.car_height // 2 + 1,
            cx - self.car_width // 2: cx + self.car_width // 2 + 1] = CAR_COLOR

        return obs

    def draw_lanes(self, obs):
        # Sidewalks
        obs[0:SIDEWALK_HEIGHT, :] = SIDEWALK_COLOR
        obs[self.screen_height - SIDEWALK_HEIGHT:self.screen_height, :] = SIDEWALK_COLOR

        # Lanes
        lane_start_y = SIDEWALK_HEIGHT
        lane_end_y = self.screen_height - SIDEWALK_HEIGHT
        lane_height = (lane_end_y - lane_start_y) // self.num_lanes

        for i in range(self.num_lanes):
            y_start = lane_start_y + i * lane_height
            y_end = y_start + lane_height
            obs[y_start:y_end, :] = ROAD_COLOR

        # Lane dividers
        for i in range(1, self.num_lanes):
            y = lane_start_y + i * lane_height
            y_start = max(0, y - DIVIDER_THICKNESS // 2)
            y_end = min(self.screen_height, y_start + DIVIDER_THICKNESS)
            for x in range(0, self.screen_width, DASH_LENGTH + GAP_LENGTH):
                x_end = min(x + DASH_LENGTH, self.screen_width)
                obs[y_start:y_end, x:x_end] = WHITE_COLOR

        # Double yellow middle line
        middle_y = lane_start_y + (lane_end_y - lane_start_y) // 2
        for x in range(0, self.screen_width, DASH_LENGTH + GAP_LENGTH):
            x_end = min(x + DASH_LENGTH, self.screen_width)
            obs[middle_y - 1:middle_y + 1, x:x_end] = YELLOW_COLOR
            obs[middle_y + SEPARATION:middle_y + SEPARATION + 2, x:x_end] = YELLOW_COLOR

    def check_collision(self):
        for car in self.cars:
            # Bounding boxes
            car_left = car['center_x'] - self.car_width // 2
            car_right = car['center_x'] + self.car_width // 2
            car_top = car['center_y'] - self.car_height // 2
            car_bottom = car['center_y'] + self.car_height // 2

            chicken_left = self.chicken_center_x - self.chicken_collision_width // 2
            chicken_right = self.chicken_center_x + self.chicken_collision_width // 2
            chicken_top = self.chicken_center_y - self.chicken_collision_height // 2
            chicken_bottom = self.chicken_center_y + self.chicken_collision_height // 2

            if (car_left < chicken_right and car_right > chicken_left and
                    car_top < chicken_bottom and car_bottom > chicken_top):
                return True
        return False

    def step(self, action):
        self.steps += 1

        old_y = self.chicken_center_y

        if action == 1:  # UP
            self.chicken_center_y = max(0, self.chicken_center_y - 1)
        elif action == 2:  # DOWN
            self.chicken_center_y = min(self.screen_height - 25, self.chicken_center_y + 1)

        # Update cars
        for car in self.cars:
            car['center_x'] += car['speed']
            if car['center_x'] > self.screen_width + 10:
                car['center_x'] = -10
                car['speed'] = random.uniform(1.5, 2.0)

        terminated = False
        truncated = False
        reward = 0

        # TODO: Old reward system? Also maybe load old semi okay model and go from there.
        # TODO: Maybe try this?
        progress = (old_y - self.chicken_center_y) / self.screen_height
        if progress > 0:
            reward += progress * 0.5

            # Check collision FIRST (before other rewards)
        if self.check_collision():
            reward = -10.0
            terminated = True
            info = {'score': self.score, 'collision': True, 'outcome': 'collision'}
            return self.get_obs(), reward, terminated, truncated, info

        # Win condition
        if self.chicken_center_y <= 10:
            reward = 10.0
            self.score += 1
            terminated = True
            info = {'score': self.score, 'collision': False, 'outcome': 'success'}
            return self.get_obs(), reward, terminated, truncated, info

        # Timeout
        if self.steps >= 5000:
            reward = -5.0
            truncated = True
            info = {'score': self.score, 'collision': False, 'outcome': 'timeout'}
            return self.get_obs(), reward, terminated, truncated, info

        # Small reward for moving forward
        # if self.chicken_center_y < old_y:
        #     reward = 0.01
        # else:
        #     reward = -0.02

        # Keep chicken within bounds (between top and bottom sidewalks)
        self.chicken_center_y = np.clip(self.chicken_center_y, 0, self.screen_height - 25)

        info = {'score': self.score, 'collision': False, 'outcome': 'ongoing'}
        return self.get_obs(), reward, terminated, truncated, info