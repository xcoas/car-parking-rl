import gymnasium as gym
import numpy as np
import math
from gymnasium import spaces
from physics import ParkingPhysics
from renderer import ParkingRenderer

class ParkingEnv(gym.Env):
    def __init__(self, initial_difficulty: float = 0.0, max_steps: int = 2000, render_mode=None, fps: int = 60, show_rays = False, show_critic_graph = False):
        super().__init__()
        self.render_mode = render_mode
        self.difficulty = initial_difficulty
        self.is_reminder = False
        self.current_step = 0
        self.action_repeat = 4
        self.max_steps = max_steps // self.action_repeat
        self.fps = fps
        self.show_rays = show_rays
        self.show_critic_graph = show_critic_graph

        self.physics = ParkingPhysics(difficulty=self.difficulty)
        if self.render_mode == 'human':
            self.renderer = ParkingRenderer(self.physics, fps=self.fps, show_rays=self.show_rays, show_critic_graph=self.show_critic_graph)

        self.action_space = spaces.Box(low=np.array([-0.5, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7 + len(self.physics.sensor_angles),), dtype=np.float32)

        self.dist_old = 0
        self.dist_new = 0

    def set_difficulty(self, level: float):
        self.difficulty = float(np.clip(level, 0.0, 1.0))

    def _get_obs(self):
        radar_readings = self.physics.get_radar_readings()

        target_x, target_y = self.physics.parking_spot_coordinate

        dx_target = target_x - self.physics.agent_body.position.x
        dy_target = target_y - self.physics.agent_body.position.y

        distance = self.get_car_distance() / math.hypot(self.physics.width, self.physics.height)
        angle_to_target = math.atan2(dy_target, dx_target)
        relative_angle = angle_to_target - self.physics.agent_body.angle
        dir_cos = math.cos(relative_angle)
        dir_sin = math.sin(relative_angle)

        sin_angle = math.sin(self.physics.agent_body.angle)
        cos_angle = math.cos(self.physics.agent_body.angle)

        vx, vy = self.physics.agent_body.velocity
        max_speed = 130.0
        norm_vx = np.clip(vx / max_speed, -1.0, 1.0)
        norm_vy = np.clip(vy / max_speed, -1.0, 1.0)

        obs = np.array([
            *radar_readings,
            distance,
            dir_sin, dir_cos,
            sin_angle, cos_angle,
            norm_vx, norm_vy
        ],dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        if self.difficulty > 0.8 and np.random.uniform(0.0, 1.0) < 0.10:
            self.is_reminder = True
            active_difficulty = round(np.random.uniform(0.1, 0.8), 1)
        else:
            self.is_reminder = False
            active_difficulty = self.difficulty

        self.physics = ParkingPhysics(difficulty=active_difficulty)
        self.physics.crashed = False

        if self.render_mode == 'human':
            self.renderer.physics = self.physics
            self.renderer.render()

        self.dist_old = self.get_car_distance()
        self.dist_new = self.get_car_distance()

        return self._get_obs(), {}

    def step(self, action):
        throttle = action[0] * 300
        steer = action[1] * 150

        terminated = False
        trunacted = False
        reward = 0

        self.dist_old = self.get_car_distance()

        for _ in range(self.action_repeat):
            self.physics.step([throttle, steer])

            if self.render_mode == 'human':
                self.renderer.render()

            if self.physics.crashed or self.physics.is_parked:
                break

        self.dist_new = self.get_car_distance()
        distance_reward_scale = max(0.0, (0.8 - self.difficulty) / 0.8)
        reward += (self.dist_old - self.dist_new) * 0.001 * distance_reward_scale

        if self.physics.crashed == True:
            terminated = True
            reward -= 10.0

        if self.physics.is_parked == True:
            terminated = True
            reward += 25.0

        if self.current_step >= self.max_steps:
            trunacted = True
            reward -= 10.0

        reward -= 0.0025 * self.action_repeat

        self.current_step += 1

        if self.render_mode == 'human':
            self.renderer.render()

        info = {"is_reminder": self.is_reminder}
        if not self.is_reminder:
            info["is_success"] = self.physics.is_parked

        return self._get_obs(), reward, terminated, trunacted, info

    def get_car_distance(self):
        dx_target = self.physics.parking_spot_coordinate[0] - self.physics.agent_body.position.x
        dy_target = self.physics.parking_spot_coordinate[1] - self.physics.agent_body.position.y

        return math.hypot(dx_target, dy_target)