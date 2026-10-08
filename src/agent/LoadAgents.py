import os
import sys
import glob
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'env')))

from env.parking_env import ParkingEnv

def load_agents(MODELS_FOLDER, DIFFICULTY, seed):
    agents = []

    for i, model_path in enumerate(MODELS_FOLDER):
        model_name = os.path.basename(model_path).replace(".zip", "")

        model_env = ParkingEnv(initial_difficulty=DIFFICULTY, render_mode=None, show_critic_graph=False, show_rays=False)
        model_loaded = PPO.load(model_path)
        model_obs, _ = model_env.reset(seed=seed)

        obj = {
            "id": i,
            "model_name": model_name,
            "model_loaded": model_loaded,
            "env": model_env,
            "obs": model_obs,
            "crashed": False,
            "parked": False,
            "alpha": 0.9,
            "steps": 0,
            "score": 0,
        }

        agents.append(obj)

    return agents