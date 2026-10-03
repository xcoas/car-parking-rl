import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'env')))

from stable_baselines3 import PPO
from env.parking_env import ParkingEnv

if __name__ == "__main__":
    env = ParkingEnv(initial_difficulty=0.9, render_mode='human')
    model = PPO.load("ppo_parking_agent")

    obs, info = env.reset()

    while True:
        action, _states = model.predict(obs, deterministic=False)

        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            critic_value = model.policy.predict_values(obs_tensor).item()

        env.renderer.critic_values.append(critic_value)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("reset")
            obs, info = env.reset()