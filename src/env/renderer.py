import pygame
import pymunk.pygame_util
import sys
import math
import os

class ParkingRenderer:
    def __init__(self, physics):
        pygame.init()

        self.physics = physics
        self.screen = pygame.display.set_mode((self.physics.width, self.physics.height))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()

        self.parking_lines = []
        spot_width = physics.SPOT_WIDTH
        spot_depth = physics.SPOT_DEPTH
        for j in range(physics.blueprint['num_of_rows'] + 1):
            x_start = ((j + 1) * physics.VERTICAL_LANE) + (j * spot_depth)
            x_end = x_start + spot_depth

            for i in range(physics.blueprint['parking_spots_per_row'] + 1):
                y = i * spot_width + physics.HORIZONTAL_LANE
                print(y)

                start_point = (x_start, y)
                end_point = (x_end, y)
                self.parking_lines.append([start_point, end_point])

        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, "assets")

        parked_car_svgs = []
        for i in range(1,4):
            img_path = os.path.join(assets_dir, f"car_parked_{i}.svg")
            parked_car_svgs.append(pygame.image.load(img_path))

        player_img_path = os.path.join(assets_dir, "player_car.svg")
        player_car_svg = pygame.image.load(player_img_path)

        self.scaled_car_svgs = []
        for car_svg in parked_car_svgs:
            scaled_svg = pygame.transform.scale(car_svg, (self.physics.PARKED_CAR_LENGTH,self.physics.PARKED_CAR_WIDTH))
            self.scaled_car_svgs.append(scaled_svg)

        self.player_car_svg = pygame.transform.scale(player_car_svg, (self.physics.PLAYER_CAR_LENGTH,self.physics.PLAYER_CAR_WIDTH))

    def render(self):
        self.screen.fill((90, 90, 90))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        for line in self.parking_lines:
            pygame.draw.line(self.screen, (250, 250, 250), line[0], line[1])

        for border in self.physics.borders:
            pygame.draw.line(self.screen, (250, 250, 250), border[0], border[1], 5)

        for car in self.physics.parked_car_bodies:
            body_car = car[0]
            sprite_idx = car[1]
            x, y = body_car.position
            angle = -math.degrees(body_car.angle)

            selected_spirte = self.scaled_car_svgs[sprite_idx - 1]
            rotated_image = pygame.transform.rotate(selected_spirte, angle)
            draw_position = rotated_image.get_rect(center=(x,y))
            self.screen.blit(rotated_image, draw_position)

        body_player = self.physics.agent_body
        sprite = self.player_car_svg
        x, y = body_player.position
        angle = -math.degrees(body_player.angle)
        rotated_image = pygame.transform.rotate(sprite, angle)
        draw_position = rotated_image.get_rect(center=(x,y))
        self.screen.blit(rotated_image, draw_position)

        pygame.display.flip()
        self.clock.tick(60)