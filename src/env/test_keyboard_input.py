import pygame
from parking_env import ParkingEnv

env = ParkingEnv(initial_difficulty=0.5, render_mode='human')
obs, info = env.reset()

while True:
    keys = pygame.key.get_pressed()

    if keys[pygame.K_w] or keys[pygame.K_UP]:
        throttle = 1
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        throttle = -0.5
    else:
        throttle = 0

    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        steer = -1
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        steer = 1
    else:
        steer = 0

    action = [throttle, steer]
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("reset")
        obs, info = env.reset()