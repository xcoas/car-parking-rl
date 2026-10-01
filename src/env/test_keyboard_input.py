import pygame
from physics import ParkingPhysics, TEST_BLUEPRINT
from renderer import ParkingRenderer

physics = ParkingPhysics(blueprint=TEST_BLUEPRINT)
renderer = ParkingRenderer(physics=physics)

while True:
    keys = pygame.key.get_pressed()

    throttle = 0
    steer = 0

    if keys[pygame.K_w] or keys[pygame.K_UP]:
        throttle = 300
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        throttle = -150

    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        steer = -150
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        steer = 150

    physics.step([throttle, steer])
    
    renderer.render()