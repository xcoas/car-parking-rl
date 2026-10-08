import sys
import os
import pygame
import numpy as np
import torch
import random
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'env')))

from stable_baselines3 import PPO
from env.parking_env import ParkingEnv
from env.physics import ParkingPhysics

from agent.LoadAgents import load_agents

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
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, 'SavedModels'))
    MODEL_FILES = glob.glob(os.path.join(MODELS_DIR, '*.zip'))

    DIFFICULTY = 1.0
    MODEL_PATH = "ppo_parking_agent"

    agent_to_compete_against = 2

    seed = random.randint(0, 1_000_000)

    agents = load_agents(MODEL_FILES, DIFFICULTY, seed)
    agent = agents[agent_to_compete_against]

    env = ParkingEnv(initial_difficulty=DIFFICULTY, render_mode=None, fps=60)
    renderer = ParkingEnv(
        initial_difficulty=DIFFICULTY, 
        render_mode='human', 
        fps=60, 
        show_critic_graph=False, 
        show_rays=False
    ).renderer

    if len(agents) > 0:
        renderer.physics = agents[agent_to_compete_against]['env'].physics

    human_score = 0
    ai_score = 0

    while True:
        round_seed = np.random.randint(0, 1_000_000)
        scores_str = f"HUMAN: {human_score} | AI: {ai_score}"

        obs, _ = agent['env'].reset(seed=round_seed)

        human_physics = ParkingPhysics(seed=round_seed, difficulty=DIFFICULTY)

        renderer.physics = agent['env'].physics
        renderer.score_text = scores_str

        round_over = False
        ai_done = False
        human_done = False

        while not round_over:
            if not ai_done:
                ai_action, _ = agent['model_loaded'].predict(obs, deterministic=True)
                obs_tensor, _ = agent['model_loaded'].policy.obs_to_tensor(obs)
                with torch.no_grad():
                    val = agent['model_loaded'].policy.predict_values(obs_tensor).item()
                    renderer.critic_values.append(val)

                ai_throttle = ai_action[0] * 300
                ai_steer = ai_action[1] * 150
            else:
                ai_throttle = 0
                ai_steer = 0

            for _ in range(agent['env'].action_repeat):
                if not ai_done:
                    agent['env'].physics.step([ai_throttle, ai_steer])
                    if agent['env'].physics.crashed or agent['env'].physics.is_parked:
                            ai_done = True

                if not human_done:
                    h_throttle, h_steer = get_human_action()
                    human_physics.step([h_throttle, h_steer])
                    if human_physics.crashed or human_physics.is_parked:
                        human_done = True

                renderer.all_agents_data = [
                    {
                        "body": agent['env'].physics.agent_body,
                        "name": f"AI ({agent['model_name']})",
                        "crashed": agent['env'].physics.crashed,
                        "parked": agent['env'].physics.is_parked,
                        "alpha": 1.0
                    },
                    {
                        "body": human_physics.agent_body,
                        "name": "YOU",
                        "crashed": human_physics.crashed,
                        "parked": human_physics.is_parked,
                        "alpha": 1.0
                    }
                ]
                renderer.render()

                if human_physics.is_parked or agent['env'].physics.is_parked or (human_done and ai_done):
                    break

            if not ai_done:
                obs = agent['env']._get_obs()

            if human_physics.is_parked:
                human_score += 1
                print("Congrats you won against AI!")
                round_over = True
                break
            elif agent['env'].physics.is_parked:
                ai_score += 1
                print("AI won this round try your luck next time!")
                round_over = True
                break
            elif human_done and ai_done:
                print("Sadly nobody of you could win this one")
                round_over = True
                break