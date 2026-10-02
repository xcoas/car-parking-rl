import gymnasium as gym
import numpy as np
import math
from gymnasium import spaces
from physics import ParkingPhysics, TEST_BLUEPRINT
from renderer import ParkingRenderer

class ParkingEnv(gym.Env):
    def __init__(self, blueprint, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.blueprint = blueprint

        self.physics = ParkingPhysics(self.blueprint)
        if self.render_mode == 'human':
            self.renderer = ParkingRenderer(self.physics)

        self.action_space = spaces.Box(low=np.array([-0.5, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7 + len(self.physics.sensor_angles),), dtype=np.float32)

        self.dist_old = 0
        self.dist_new = 0

        self.current_step = 0
        self.max_steps = 2000
        dx_target = dy_target = 0

    def _get_obs(self):
        radar_readings = self.physics.get_radar_readings()

        self.parking_spot_coordinate = self.physics.calculate_spot_coordinates([self.blueprint['car_target_spot']])[0]

        dx_target = self.parking_spot_coordinate[0] - self.physics.agent_body.position.x
        dy_target = self.parking_spot_coordinate[1] - self.physics.agent_body.position.y

        distance = self.get_car_distance() / math.hypot(self.physics.width, self.physics.height)
        angle_to_target = math.atan2(dy_target, dx_target)
        relative_angle = angle_to_target - self.physics.agent_body.angle
        dir_cos = math.cos(relative_angle)
        dir_sin = math.sin(relative_angle)

        sin_angle = math.sin(self.physics.agent_body.angle)
        cos_angle = math.cos(self.physics.agent_body.angle)

        vx, vy = self.physics.agent_body.velocity

        obs = np.array([
            *radar_readings,
            distance,
            dir_sin, dir_cos,
            sin_angle, cos_angle,
            vx, vy
        ],dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.physics = ParkingPhysics(self.blueprint)
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

        self.physics.step([throttle, steer])

        self.dist_new = self.get_car_distance()
        reward += (self.dist_old - self.dist_new) * 0.05

        if self.physics.crashed == True:
            terminated = True
            reward -= 3.0

        if self.physics.is_parked == True:
            terminated = True
            reward += 10.0

        if self.current_step >= self.max_steps:
            trunacted = True
            reward -= 1.0

        reward -= 0.001

        self.current_step += 1

        if self.render_mode == 'human':
            self.renderer.render()

        return self._get_obs(), reward, terminated, trunacted, {}

    def get_car_distance(self):
        dx_target = self.physics.parking_spot_coordinate[0] - self.physics.agent_body.position.x
        dy_target = self.physics.parking_spot_coordinate[1] - self.physics.agent_body.position.y

        return math.hypot(dx_target, dy_target)