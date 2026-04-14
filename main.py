import pygame
import sys

from env.freeway_env import FreewayENV
from graphics import Graphics
from neuralnet.dqn_player import DQNPlayer

pygame.init()
pygame.font.init()

# Adjust fonts as necessary, falling back to a default system font
FONT = pygame.font.SysFont("Arial", 36)
TITLE_FONT = pygame.font.SysFont("Arial", 60, bold=True)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)


def draw_text(surface, text, font, color, x, y, center=True):
    """Helper function to render text to the screen."""
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect()
    if center:
        text_rect.center = (x, y)
    else:
        text_rect.topleft = (x, y)
    surface.blit(text_obj, text_rect)


def show_main_menu(screen):
    """Displays the main menu and waits for a mode selection."""
    screen.fill(BLACK)

    center_x = screen.get_width() // 2
    draw_text(screen, "FREEWAY AI", TITLE_FONT, WHITE, center_x, 150)
    draw_text(screen, "Press 1 - Manual Play", FONT, WHITE, center_x, 250)
    draw_text(screen, "Press 2 - AI Play", FONT, WHITE, center_x, 320)
    draw_text(screen, "Press Q - Quit", FONT, RED, center_x, 400)

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "MANUAL"
                elif event.key == pygame.K_2:
                    return "AI"
                elif event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()


def show_game_over(screen, info, total_reward):
    """Displays the death/win screen and handles restart."""
    screen.fill(BLACK)

    center_x = screen.get_width() // 2

    if info["outcome"] == "success":
        title_text = "WIN"
        title_color = GREEN
    else:
        title_text = "LOSE"
        title_color = RED

    draw_text(screen, title_text, TITLE_FONT, title_color, center_x, 80)

    draw_text(screen, "Press R - Play Again", FONT, GREEN, center_x, 170)
    draw_text(screen, "Press M - Main Menu", FONT, WHITE, center_x, 230)

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return "REPLAY"
                elif event.key == pygame.K_m:
                    return "MENU"


def play_manual(env, graphics):
    """Human-controlled loop."""
    observation, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    action_map = {
        pygame.K_w: 1, pygame.K_UP: 1,
        pygame.K_s: 2, pygame.K_DOWN: 2,
    }

    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action = 0
        keys = pygame.key.get_pressed()
        for key, act in action_map.items():
            if keys[key]:
                action = act
                break
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        graphics.render(env)

    return info, total_reward


def play_ai(env, graphics, nn_player):
    """AI-controlled loop."""
    observation, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    nn_player.reset(observation)

    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action = nn_player.get_action(observation)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        graphics.render(env)

    return info, total_reward


def main():
    env = FreewayENV()
    graphics = Graphics()

    graphics.render(env)

    screen = pygame.display.get_surface()

    if screen is None:
        raise RuntimeError("Graphics class did not initialize a pygame display surface.")

    nn_player = DQNPlayer(env.action_space.n, "some_model_6.72.pth")

    current_state = "MENU"
    selected_mode = None
    last_info = None
    last_reward = 0

    while True:
        if current_state == "MENU":
            selected_mode = show_main_menu(screen)
            current_state = "PLAY"

        elif current_state == "PLAY":
            if selected_mode == "MANUAL":
                last_info, last_reward = play_manual(env, graphics)
            elif selected_mode == "AI":
                last_info, last_reward = play_ai(env, graphics, nn_player)

            current_state = "GAME_OVER"

        elif current_state == "GAME_OVER":
            user_choice = show_game_over(screen, last_info, last_reward)

            if user_choice == "REPLAY":
                current_state = "PLAY"
            elif user_choice == "MENU":
                current_state = "MENU"


if __name__ == "__main__":
    main()