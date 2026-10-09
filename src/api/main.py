import sys
import os
import random
import numpy as np
import glob
import torch
from fastapi import FastAPI, HTTPException

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
ENV_DIR = os.path.abspath(os.path.join(SRC_DIR, 'env'))
AGENT_DIR = os.path.abspath(os.path.join(SRC_DIR, 'agent'))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))
MODELS_DIR = os.path.abspath(os.path.join(SRC_DIR, 'SavedModels'))
MODEL_FILES = glob.glob(os.path.join(MODELS_DIR, '*.zip'))

sys.path.extend([CURRENT_DIR, SRC_DIR, ENV_DIR, AGENT_DIR, ROOT_DIR])

from stable_baselines3 import PPO
from env.parking_env import ParkingEnv
from env.physics import ParkingPhysics
from agent.LoadAgents import load_agents
from schemas import (
    ModelInfoResponse,
    Simulation_Request,
    SimulationResponse,
    CompareRequest,
    ModelStats,
    CompareResponse
)

app = FastAPI(
    title="Autonomous Parking RL API",
    description="API for testing and comparing trained PPO in ParkingEnv environment",
    version="1.0.0"
)

def find_model_path(filename: str) -> str:
    for folder in [ROOT_DIR, AGENT_DIR, CURRENT_DIR]:
        candidate = os.path.join(folder, filename)
        if os.path.exists(candidate):
            return candidate
    return filename

MODELS = {}
try:
    agents = load_agents(MODEL_FILES, DIFFICULTY=1.0, seed=42)

    for agent in agents:
        MODELS[agent['model_name']] = agent['model_loaded']
except Exception as e:
    print(f"failed to load models: {e}")

def get_critic_value(model: PPO, obs: np.ndarray) -> float:
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        return round(float(model.policy.predict_values(obs_tensor).item()), 2)

def get_model_arch(model: PPO) -> dict:
    actor_layers = [
        layer.out_features
        for layer in model.policy.mlp_extractor.policy_net
        if isinstance(layer, torch.nn.Linear)
    ]
    critic_layers = [
        layer.out_features 
        for layer in model.policy.mlp_extractor.value_net 
        if isinstance(layer, torch.nn.Linear)
    ]
    return {"actor": actor_layers, "critic": critic_layers}

def run_single_episode(model: PPO, difficulty: float, seed: int) -> dict:
    game_model = model
    difficulty = difficulty
    seed = seed

    return_episode = {
        'crashed': False,
        'is_success': False,
        'steps_taken': 0,
        'initial_critic_value': 0,
        'final_critic_value': 0,
    }

    env = ParkingEnv(initial_difficulty=difficulty, render_mode=None)

    random.seed(seed)
    np.random.seed(seed)
    obs, _states = env.reset(seed=seed)

    if getattr(env, 'is_reminder', False):
        random.seed(seed)
        np.random.seed(seed)
        env.physics = ParkingPhysics(difficulty=difficulty)
        obs = env._get_obs()

    return_episode['initial_critic_value'] = get_critic_value(model = game_model, obs = obs)

    for i in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)

        throttle = action[0] * 300
        steer = action[1] * 150

        for _ in range(env.action_repeat):
            env.physics.step([throttle, steer])
            if env.physics.crashed or env.physics.is_parked:
                break

        obs = env._get_obs()
        return_episode['steps_taken'] = i + 1

        if env.physics.crashed:
            return_episode['crashed'] = True
            return_episode['final_critic_value'] = get_critic_value(model = game_model, obs = obs)
            break
        elif env.physics.is_parked:
            return_episode['is_success'] = True
            return_episode['final_critic_value'] = get_critic_value(model = game_model, obs = obs)
            break
        elif i >= env.max_steps - 1:
            return_episode['final_critic_value'] = get_critic_value(model = game_model, obs = obs)
            break

    return_episode['final_critic_value'] = get_critic_value(model=model, obs=obs)

    return return_episode

@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded")

    first_model = list(MODELS.values())[0]

    return ModelInfoResponse(
        available_models = list(MODELS.keys()),
        observation_space_size = first_model.observation_space.shape[0],
        lidar_rays = {name: model.observation_space.shape[0] - 7 for name, model in MODELS.items()},
        action_space = ['throttle', 'steer'],
        network_architecture = {name: get_model_arch(model) for name, model in MODELS.items()}
    )

@app.post("/simulate", response_model=SimulationResponse)
def simulate(request: Simulation_Request):
    if request.model_version not in MODELS:
        raise HTTPException(status_code=400, detail="wrong model version")

    if request.seed is not None:
        used_seed = request.seed
    else:
        used_seed = random.randint(0, 1_000_000)

    result = run_single_episode(
        model=MODELS[request.model_version], 
        difficulty=request.difficulty, 
        seed=used_seed
    )

    response = SimulationResponse(
        model_used=request.model_version,
        difficulty=request.difficulty,
        seed=used_seed,
        is_success=result['is_success'],
        crashed=result['crashed'],
        steps_taken=result['steps_taken'],
        initial_critic_value=result['initial_critic_value'],
        final_critic_value=result['final_critic_value']
    )

    return response

@app.post("/compare", response_model=CompareResponse)
def compare_models(request: CompareRequest):
    keys = list(MODELS.keys())
    if len(keys) < 2:
        raise HTTPException(
            status_code=503, 
            detail="You need atleast 2 models to use compare"
        )

    wins = {name: 0 for name in keys}
    crashes = {name: 0 for name in keys}
    timesteps = {name: 0 for name in keys}
    stats = {}

    for _ in range(request.episodes):
        seed = random.randint(0, 1_000_000)

        for model_name in keys:
            result = run_single_episode(
                MODELS[model_name], 
                difficulty=request.difficulty, 
                seed=seed
            )

            if result['is_success']:
                wins[model_name] += 1
                timesteps[model_name] += result['steps_taken']
            if result['crashed']:
                crashes[model_name] += 1

    highest_winrate = -1.0
    winner_name = 'tie'
    is_tie = False

    for model_name in keys:
        winrate = round((wins[model_name] / request.episodes) * 100.0, 2) if request.episodes > 0 else 0.0
        avg_steps = round((timesteps[model_name] / wins[model_name]), 2) if wins[model_name] > 0 else 0.0
        
        stats[model_name] = ModelStats(
            win_rate_percent=winrate, 
            crashes=crashes[model_name], 
            avg_steps_on_success=avg_steps
        )

        if winrate > highest_winrate:
            highest_winrate = winrate
            winner_name = model_name
            is_tie = False
        elif winrate == highest_winrate:
            is_tie = True

    if is_tie:
        winner_name = 'tie'

    return CompareResponse(
        difficulty=request.difficulty,
        episodes_played=request.episodes,
        winner=winner_name,
        model_stats=stats
    )