import sys
import os
import pygame
import numpy as np
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'env')))

from stable_baselines3 import PPO
from env.parking_env import ParkingEnv
from env.physics import ParkingPhysics

if __name__ == "__main__":
    DIFFICULTY = 0.6

    MODEL_1_PATH = "./ppo_parking_agent (modelv0).zip"
    MODEL_2_PATH = "./ppo_parking_agent (modelv1).zip"

    env_1 = ParkingEnv(initial_difficulty=DIFFICULTY, render_mode=None, show_critic_graph=False, show_rays=False)
    env_2 = ParkingEnv(initial_difficulty=DIFFICULTY, render_mode=None, show_critic_graph=False, show_rays=False)
    human_physics = ParkingPhysics(difficulty=DIFFICULTY)
    model_1 = PPO.load(MODEL_1_PATH)
    model_2 = PPO.load(MODEL_2_PATH)

    renderer = ParkingEnv(
        initial_difficulty=DIFFICULTY, 
        render_mode='human', 
        fps=60, 
        show_critic_graph=False, 
        show_rays=False
    ).renderer

    renderer.agent_label = "v0"
    renderer.human_label = "v1"
    renderer.agent_alpha = 1.0
    renderer.human_alpha = 1.0

    score_1 = 0
    score_2 = 0

    while True:
        seed = random.randint(0, 1_000_000)

        random.seed(seed)
        np.random.seed(seed)
        obs_1, _ = env_1.reset(seed=seed)
        if env_1.is_reminder:
            random.seed(seed)
            np.random.seed(seed)
            env_1.physics = ParkingPhysics(difficulty=DIFFICULTY)
            obs_1 = env_1._get_obs()

        random.seed(seed)
        np.random.seed(seed)
        obs_2, _ = env_2.reset(seed=seed)
        if env_2.is_reminder:
            random.seed(seed)
            np.random.seed(seed)
            env_2.physics = ParkingPhysics(difficulty=DIFFICULTY)
            obs_2 = env_2._get_obs()

        renderer.physics = env_1.physics
        renderer.human_body = env_2.physics.agent_body
        renderer.score_text = f"Model v0: {score_1}   |   Model v1: {score_2}  (Diff: {DIFFICULTY})"

        done_1 = False
        done_2 = False
        round_over = False

        while not round_over:
            if not done_1:
                action_1, _ = model_1.predict(obs_1, deterministic=False)
            else:
                action_1 = [0.0, 0.0]

            if not done_2:
                action_2, _ = model_2.predict(obs_2, deterministic=False)
            else:
                action_2 = [0.0, 0.0]

            throttle_1 = action_1[0] * 300
            steer_1 = action_1[1] * 150

            throttle_2 = action_2[0] * 300
            steer_2 = action_2[1] * 150

            if not done_1:
                env_1.physics.step([throttle_1, steer_1])
                if env_1.physics.crashed or env_1.physics.is_parked:
                    done_1 = True

            if not done_2:
                env_2.physics.step([throttle_2, steer_2])
                if env_2.physics.crashed or env_2.physics.is_parked:
                    done_2 = True

            renderer.render()

            if env_2.physics.is_parked:
                score_2 += 1
                print("Model v1 won")
                round_over = True
                break
            elif env_1.physics.is_parked:
                score_1 += 1
                print("Model v0 won")
                round_over = True
                break
            elif done_1 and done_2:
                print("Nobody won")
                round_over = True
                break

            if not done_1:
                obs_1 = env_1._get_obs()
            if not done_2:
                obs_2 = env_2._get_obs()
