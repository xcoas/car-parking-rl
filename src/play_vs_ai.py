import sys
import os
import pygame
import numpy as np
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'env')))

from stable_baselines3 import PPO
from env.parking_env import ParkingEnv
from env.physics import ParkingPhysics

def get_human_action():
    keys = pygame.key.get_pressed()
    throttle = 0.0
    steer = 0.0

    if keys[pygame.K_UP] or keys[pygame.K_w]:
        throttle = 1.0
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        throttle = -0.5

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        steer = -1.0
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        steer = 1.0

    return [throttle * 300, steer * 150]

if __name__ == "__main__":
    DIFFICULTY = 0.5
    MODEL_PATH = "ppo_parking_agent"

    env = ParkingEnv(initial_difficulty=DIFFICULTY, render_mode=None, fps=60)
    renderer = ParkingEnv(initial_difficulty=DIFFICULTY, render_mode='human', fps=60).renderer

    model = PPO.load(MODEL_PATH)

    human_score = 0
    ai_score = 0

    while True:
        round_seed = np.random.randint(0, 1_000_000)

        np.random.seed(round_seed)
        random.seed(round_seed)
        obs, _ = env.reset(seed=round_seed)

        np.random.seed(round_seed)
        random.seed(round_seed)
        human_physics = ParkingPhysics(difficulty=DIFFICULTY)

        renderer.physics = env.physics
        renderer.human_body = human_physics.agent_body
        renderer.critic_values.clear()
        renderer.score_text = f"ME: {human_score}   |   AI: {ai_score}  (Difficulty: {DIFFICULTY})"

        round_over = False
        ai_done = False
        human_done = False

        while not round_over:
            if not ai_done:
                ai_action, _ = model.predict(obs, deterministic=True)
                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    val = model.policy.predict_values(obs_tensor).item()
                    renderer.critic_values.append(val)
            else:
                ai_action = [0.0, 0.0]

            ai_throttle = ai_action[0] * 300
            ai_steer = ai_action[1] * 150

            if not ai_done:
                env.physics.step([ai_throttle, ai_steer])
                if env.physics.crashed or env.physics.is_parked:
                    ai_done = True

            if not human_done:
                h_throttle, h_steer = get_human_action()
                human_physics.step([h_throttle, h_steer])
                if human_physics.crashed or human_physics.is_parked:
                    human_done = True

            renderer.render()

            if human_physics.is_parked:
                human_score += 1
                print("Congrats you won against AI!")
                round_over = True
                break
            elif env.physics.is_parked:
                ai_score += 1
                print("AI won this round try your luck next time!")
                round_over = True
                break
            elif human_done and ai_done:
                print("Sadly nobody of you could win this one")
                round_over = True
                break

            if not ai_done:
                obs = env._get_obs()