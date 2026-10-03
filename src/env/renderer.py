import pygame
import pymunk.pygame_util
import sys
import math
import os
from collections import deque

class ParkingRenderer:
    def __init__(self, physics, fps: int = 60):
        pygame.init()

        self.physics = physics
        self.fps = fps
        self.screen = pygame.display.set_mode((self.physics.width, self.physics.height))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()

        spot_width = physics.SPOT_WIDTH
        spot_depth = physics.SPOT_DEPTH

        self.human_body = None
        self.score_text = ""

        self.critic_values = deque(maxlen=100)
        self.font = pygame.font.SysFont("Arial", 16, bold=True)

        target_row, target_spot = physics.blueprint['car_target_spot']
        target_x = ((target_row + 1) * physics.VERTICAL_LANE) + (target_row * spot_depth)
        target_y = physics.HORIZONTAL_LANE + (target_spot * spot_width)
        self.target_rect = pygame.Rect(target_x, target_y, spot_depth, spot_width)

        self.parking_lines = []
        spot_width = physics.SPOT_WIDTH
        spot_depth = physics.SPOT_DEPTH
        for j in range(physics.blueprint['num_of_rows']):
            x_start = ((j + 1) * physics.VERTICAL_LANE) + (j * spot_depth)
            x_end = x_start + spot_depth

            for i in range(physics.blueprint['parking_spots_per_row'] + 1):
                y = i * spot_width + physics.HORIZONTAL_LANE

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

        target_row, target_spot = self.physics.blueprint['car_target_spot']
        target_x = ((target_row + 1) * self.physics.VERTICAL_LANE) + (target_row * self.physics.SPOT_DEPTH)
        target_y = self.physics.HORIZONTAL_LANE + (target_spot * self.physics.SPOT_WIDTH)
        target_rect = pygame.Rect(target_x, target_y, self.physics.SPOT_DEPTH, self.physics.SPOT_WIDTH)
        
        pygame.draw.rect(self.screen, (0, 100, 0), target_rect)

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

        for start_pos, end_pos, is_hit in self.physics.radar_rays:
            color = (255, 60, 60) if is_hit else (60, 255, 60)
            pygame.draw.line(self.screen, color, start_pos, end_pos)

            if is_hit:
                pygame.draw.circle(self.screen, (255, 0, 0), (int(end_pos[0]), int(end_pos[1])), 4)

        body_player = self.physics.agent_body
        sprite = self.player_car_svg
        x, y = body_player.position
        angle = -math.degrees(body_player.angle)
        rotated_image = pygame.transform.rotate(sprite, angle)
        draw_position = rotated_image.get_rect(center=(x,y))
        self.screen.blit(rotated_image, draw_position)

        if self.human_body is not None:
            hx, hy = self.human_body.position
            h_angle = -math.degrees(self.human_body.angle)
            h_rotated = pygame.transform.rotate(self.player_car_svg, h_angle)
            h_rect = h_rotated.get_rect(center=(hx, hy))
            self.screen.blit(h_rotated, h_rect)

            ai_tag = self.font.render("AI", True, (80, 180, 255))
            human_tag = self.font.render("TY", True, (80, 255, 80))
            self.screen.blit(ai_tag, (int(x) - 10, int(y) - 35))
            self.screen.blit(human_tag, (int(hx) - 10, int(hy) - 35))

        if self.score_text:
            score_surf = self.font.render(self.score_text, True, (255, 255, 255), (35, 35, 35))
            score_rect = score_surf.get_rect(center=(self.physics.width // 2, 25))
            self.screen.blit(score_surf, score_rect)

        if len(self.critic_values) > 1:
            graph_x, graph_y = 15, 15
            graph_w, graph_h = 200, 90
            min_val, max_val = -25.0, 25.0

            bg_rect = pygame.Rect(graph_x, graph_y, graph_w, graph_h)
            pygame.draw.rect(self.screen, (35, 35, 35), bg_rect)

            zero_y = graph_y + (graph_h // 2)
            pygame.draw.line(self.screen, (90, 90, 90), (graph_x, zero_y), (graph_x + graph_w, zero_y), 1)

            points = []
            for i, val in enumerate(self.critic_values):
                px = graph_x + int((i / 99.0) * graph_w)

                clamped_val = max(min_val, min(max_val, val))
                ratio = (clamped_val - min_val) / (max_val - min_val)
                py = (graph_y + graph_h) - int(ratio * graph_h)
                points.append((px, py))

            last_val = self.critic_values[-1]
            line_color = (50, 255, 50) if last_val >= 0 else (255, 70, 70)
            pygame.draw.lines(self.screen, line_color, False, points, 2)

            label = self.font.render(f"Critic: {last_val:+.1f}", True, line_color)
            self.screen.blit(label, (graph_x + 6, graph_y + 4))

        pygame.display.flip()
        self.clock.tick(self.fps)