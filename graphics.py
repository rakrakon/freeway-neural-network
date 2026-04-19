import pygame
import numpy as np
from scipy import ndimage

from utils.config_utils import load_config

CHICKEN_SPRITE_PATH = "assets/chicken.png"
CAR_SPRITE_PATH = "assets/cars/red_car.png"

WHITE_COLOR = [255, 255, 255]
YELLOW_COLOR = [255, 255, 0]
ROAD_COLOR = [118, 122, 122]
SIDEWALK_COLOR = [180, 180, 180]
DASH_WIDTH = 10

class Graphics:
    def __init__(self):
        # Initialize pygame FIRST
        pygame.init()

        self.config = load_config()
        self.screen_width = self.config.graphics.width
        self.screen_height = self.config.graphics.height

        self.num_lanes = self.config.env.num_lanes

        # Set display mode BEFORE loading images
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Freeway - Human Player")

        # NOW load the sprites
        self.car_sprite = pygame.image.load(CAR_SPRITE_PATH).convert_alpha()
        self.chicken_sprite = pygame.image.load(CHICKEN_SPRITE_PATH).convert_alpha()

        # Initialize font for score display
        self.font = pygame.font.SysFont("pixelmix", 28, bold=True)

        self.clock = pygame.time.Clock()

    def _draw_environment(self, scale):
        """Draws the road surface, sidewalks, and lane markings."""
        road_top = 9 * scale
        road_bottom = 79 * scale
        road_height = road_bottom - road_top

        # Top sidewalk
        pygame.draw.rect(self.screen, SIDEWALK_COLOR, (0, 0, self.screen_width, road_top))

        # Bottom sidewalk
        pygame.draw.rect(self.screen, SIDEWALK_COLOR,
                         (0, road_bottom, self.screen_width, self.screen_height - road_bottom))

        # Road
        pygame.draw.rect(self.screen, ROAD_COLOR, (0, road_top, self.screen_width, road_height))

        lane_spacing = road_height // self.num_lanes


        for i in range(1, self.num_lanes):
            line_y = road_top + (i * lane_spacing)
            # Lane separators
            for x in range(0, self.screen_width, DASH_WIDTH * 2):
                pygame.draw.line(self.screen, WHITE_COLOR, (x, line_y), (x + DASH_WIDTH, line_y), 3)

    def _process_obs(self, obs):
        # Extract cars (red pixels)
        red_mask = (obs[:, :, 0] == 255) & (obs[:, :, 1] == 0) & (obs[:, :, 2] == 0)

        # Extract chicken (green pixels)
        green_mask = (obs[:, :, 0] == 0) & (obs[:, :, 1] == 255) & (obs[:, :, 2] == 0)

        cars = []

        if red_mask.any():
            labeled, num_features = ndimage.label(red_mask)
            for i in range(1, num_features + 1):
                positions = np.argwhere(labeled == i)
                if len(positions) > 0:
                    min_y = positions[:, 0].min()
                    min_x = positions[:, 1].min()
                    cars.append((min_x, min_y))

        chicken = None
        if green_mask.any():
            chicken_positions = np.argwhere(green_mask)
            min_y = chicken_positions[:, 0].min()
            min_x = chicken_positions[:, 1].min()
            chicken = (min_x, min_y)

        return {
            'cars': cars,
            'chicken': chicken
        }

    def render(self, obs, score: int):
        """Main render loop."""
        self.screen.fill((0, 0, 0))

        scale = self.screen_width // 84

        # Draw background
        self._draw_environment(scale)

        # Process & draw Entities
        entities = self._process_obs(obs)

        # Scale & blit sprites
        scaled_car = pygame.transform.scale(self.car_sprite, (6 * scale, 3 * scale))
        scaled_chicken = pygame.transform.scale(self.chicken_sprite, (3 * scale, 3 * scale))

        for car_x, car_y in entities['cars']:
            self.screen.blit(scaled_car, (car_x * scale, car_y * scale))

        if entities['chicken'] is not None:
            cx, cy = entities['chicken']
            self.screen.blit(scaled_chicken, (cx * scale, cy * scale))

        # Draw score
        score_text = self.font.render(f"Score: {score}", True, WHITE_COLOR)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)