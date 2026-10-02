import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'env')))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env.physics import TEST_BLUEPRINT
from env.parking_env import ParkingEnv

env = ParkingEnv(blueprint=TEST_BLUEPRINT, render_mode='human')
model = PPO.load("ppo_parking_agent")

obs, info = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("reset")
        obs, info = env.reset()