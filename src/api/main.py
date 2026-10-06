import sys
import os
import random
import numpy as np
import torch
from fastapi import FastAPI, HTTPException

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
ENV_DIR = os.path.abspath(os.path.join(SRC_DIR, 'env'))
AGENT_DIR = os.path.abspath(os.path.join(SRC_DIR, 'agent'))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))

sys.path.extend([CURRENT_DIR, SRC_DIR, ENV_DIR, AGENT_DIR, ROOT_DIR])

from stable_baselines3 import PPO
from env.parking_env import ParkingEnv
from env.physics import ParkingPhysics
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
    MODELS["v0"] = PPO.load(find_model_path("ppo_parking_agent (modelv0).zip"))
    MODELS["v1"] = PPO.load(find_model_path("ppo_parking_agent (modelv1).zip"))
    print("model v0 and v1 successfuly loaded")
except Exception as e:
    print(f"failed to load models: {e}")

def get_critic_value(model: PPO, obs: np.ndarray) -> float:
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        return round(float(model.policy.predict_values(obs_tensor).item()), 2)

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
    if "v0" not in MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return ModelInfoResponse(
        available_models = list(MODELS.keys()),
        observation_space_size = MODELS['v0'].observation_space.shape[0],
        lidar_rays = {name: model.observation_space.shape[0] - 7 for name, model in MODELS.items()},
        action_space = ['throttle', 'steer'],
        network_architecture = {"actor": [256,256,128], "critic": [512,256,128]},
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
    if "v0" not in MODELS or "v1" not in MODELS:
        raise HTTPException(status_code=503, detail="there is no v0 and v1 loaded models")

    v0_wins, v0_crashes, v0_timesteps = 0, 0, 0
    v1_wins, v1_crashes, v1_timesteps = 0, 0, 0

    for _ in range(request.episodes):
        seed = random.randint(0, 1_000_000)

        v0_return = run_single_episode(MODELS['v0'], difficulty=request.difficulty, seed=seed)
        v1_return = run_single_episode(MODELS['v1'], difficulty=request.difficulty, seed=seed)

        if v0_return['is_success'] == True:
            v0_wins += 1
            v0_timesteps += v0_return['steps_taken']
        if v1_return['is_success'] == True:
            v1_wins += 1
            v1_timesteps += v1_return['steps_taken']

        if v0_return['crashed'] == True:
            v0_crashes += 1
        if v1_return['crashed'] == True:
            v1_crashes += 1

    v0_winrate = round((v0_wins / request.episodes) * 100.0, 2) if request.episodes > 0 else 0.0
    v1_winrate = round((v1_wins / request.episodes) * 100.0, 2) if request.episodes > 0 else 0.0

    v0_avg_steps = round((v0_timesteps / v0_wins), 2) if v0_wins > 0 else 0.0
    v1_avg_steps = round((v1_timesteps / v1_wins), 2) if v1_wins > 0 else 0.0

    stats_v0 = ModelStats(win_rate_percent=v0_winrate, crashes=v0_crashes, avg_steps_on_success=v0_avg_steps)
    stats_v1 = ModelStats(win_rate_percent=v1_winrate, crashes=v1_crashes, avg_steps_on_success=v1_avg_steps)

    winner = ''

    if stats_v0.win_rate_percent > stats_v1.win_rate_percent:
        winner = 'v0'
    elif stats_v0.win_rate_percent < stats_v1.win_rate_percent:
        winner = 'v1'
    else:
        winner = 'tie'

    return CompareResponse(
        difficulty=request.difficulty,
        episodes_played=request.episodes,
        winner=winner,
        model_v0_stats=stats_v0,
        model_v1_stats=stats_v1
    )