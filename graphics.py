import pygame

from utils.config_utils import load_config

CHICKEN_SPRITE_PATH = "assets/chicken.png"
CAR_SPRITE_PATH = "assets/cars/red_car.png"

class Graphics:
    def __init__(self):
        self.config = load_config()
        self.screen_width = self.config['env']['width']
        self.screen_height = self.config['env']['height']
        self.car_sprite = None
        self.chicken_sprite = None
        self.top_padding = None
        self.bottom_padding = None
        self.left_padding = None
        self.right_padding = None
        self.screen = None
        self.padding = 20
        self.font = None

    def render(self, env):
        if self.screen is None:
            pygame.init()
            display_w, display_h = env.screen_width, env.screen_height
            self.screen = pygame.display.set_mode((display_w, display_h))
            pygame.display.set_caption("Freeway - Human Player")
            env.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 28)

            self.top_padding = display_h // self.padding
            self.bottom_padding = self.top_padding
            self.left_padding = display_w // self.padding
            self.right_padding = self.left_padding

            self.chicken_sprite = pygame.image.load(CHICKEN_SPRITE_PATH).convert_alpha()
            self.car_sprite = pygame.image.load(CAR_SPRITE_PATH).convert_alpha()
            self.chicken_sprite = pygame.transform.scale(self.chicken_sprite, (16, 16))
            self.car_sprite = pygame.transform.scale(self.car_sprite, (32, 16))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        obs = env.get_obs()

        draw_w = self.screen_width - self.left_padding - self.right_padding
        draw_h = self.screen_height - self.top_padding - self.bottom_padding

        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1)).convert()
        surf = pygame.transform.scale(surf, (draw_w, draw_h))

        self.screen.fill((0, 0, 0))
        self.screen.blit(surf, (self.left_padding, self.top_padding))

        scale_x = draw_w / env.screen_width
        scale_y = draw_h / env.screen_height

        # Draw cars
        for car in env.cars:
            px = int(car['center_x'] * scale_x + self.left_padding)
            py = int(car['center_y'] * scale_y + self.top_padding)
            car_rect = self.car_sprite.get_rect(center=(px, py))
            self.screen.blit(self.car_sprite, car_rect)

        # Draw chicken
        px = int(env.chicken_center_x * scale_x + self.left_padding)
        py = int(env.chicken_center_y * scale_y + self.top_padding)
        chicken_rect = self.chicken_sprite.get_rect(center=(px, py))
        self.screen.blit(self.chicken_sprite, chicken_rect)

        pygame.display.flip()
        env.clock.tick(env.fps)
