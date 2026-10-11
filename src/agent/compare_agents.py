import sys
import os
import glob
import pygame
import numpy as np
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'env')))

from stable_baselines3 import PPO
from env.parking_env import ParkingEnv
from env.physics import ParkingPhysics

from LoadAgents import load_agents

if __name__ == "__main__":
    DIFFICULTY = 1.0

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
    MODELS_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'SavedModels'))
    MODEL_FILES = glob.glob(os.path.join(MODELS_DIR, '*.zip'))

    seed = random.randint(0, 1_000_000)

    agents = load_agents(MODEL_FILES, DIFFICULTY, seed)

    print("successfully loaded all models")

    renderer = ParkingEnv(
        initial_difficulty=DIFFICULTY, 
        render_mode='human', 
        fps=60, 
        show_critic_graph=False, 
        show_rays=False
    ).renderer

    if len(agents) > 0:
        renderer.physics = agents[0]['env'].physics

    while True:
        seed = random.randint(0, 1_000_000)
        scores_str = " | ".join([f"{a['model_name']}: {a['score']}" for a in agents])
        renderer.score_text = f"Wyniki: {scores_str}  (Diff: {DIFFICULTY})"

        for agent in agents:
            random.seed(seed)
            np.random.seed(seed)
            
            obs, _ = agent['env'].reset(seed=seed)
            agent['crashed'] = False
            agent['parked'] = False
            agent['steps'] = 0

        if len(agents) > 0:
            renderer.physics = agents[0]['env'].physics
            renderer.update_blueprint(agents[0]['env'].physics.blueprint)

        round_over = False

        while not round_over:
            active_agents = [a for a in agents if not a['crashed'] and not a['parked']]
            
            if not active_agents:
                break

            for agent in active_agents:
                action, _ = agent['model_loaded'].predict(agent['obs'], deterministic=True)
                agent['current_action'] = action

            action_repeat = agents[0]['env'].action_repeat
            
            for _ in range(action_repeat):
                for agent in active_agents:
                    if agent['crashed'] or agent['parked']:
                        continue
                        
                    throttle = agent['current_action'][0] * 300
                    steer = agent['current_action'][1] * 150
                    
                    agent['env'].physics.step([throttle, steer])
                    
                    if agent['env'].physics.crashed:
                        agent['crashed'] = True
                    elif agent['env'].physics.is_parked:
                        agent['parked'] = True
                        agent['score'] += 1

                renderer.all_agents_data = [
                    {
                        "body": a['env'].physics.agent_body, 
                        "name": str(a['model_name']),
                        "crashed": a['crashed'],
                        "parked": a['parked'],
                        "alpha": a['alpha']
                    } 
                    for a in agents
                ]
                renderer.render()

            for agent in active_agents:
                if not agent['crashed'] and not agent['parked']:
                    agent['obs'] = agent['env']._get_obs()
                    agent['steps'] += 1
                    agent['env'].current_step += 1

            if round_over:
                break

            renderer.all_agents_data = [
                {
                    "body": a['env'].physics.agent_body, 
                    "name": str(a['model_name']),
                    "crashed": a['crashed'],
                    "parked": a['parked'],
                    "alpha": a['alpha']
                } 
                for a in agents
            ]
            renderer.render()
