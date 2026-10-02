import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'env')))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env.physics import TEST_BLUEPRINT
from env.parking_env import ParkingEnv

env = ParkingEnv(blueprint=TEST_BLUEPRINT)

check_env(env)
print("ParkingEnv passed validation!")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)

model.save("ppo_parking_agent")
print("model zapisany")