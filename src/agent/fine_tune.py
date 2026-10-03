import sys
import os
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'env')))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env.parking_env import ParkingEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from train import DifficultyCallback

if __name__ == "__main__":
    START_DIFFICULTY = 0.5

    test_env = ParkingEnv(initial_difficulty=START_DIFFICULTY)
    check_env(test_env)
    print("ParkingEnv passed validation!")

    env = make_vec_env(lambda: ParkingEnv(initial_difficulty=START_DIFFICULTY), n_envs=4)

    model = PPO.load(
        "ppo_parking_agent", 
        env=env,
        custom_objects={"learning_rate": 0.00001}
    )
    
    difficulty_callback = DifficultyCallback(current_difficulty=START_DIFFICULTY)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints_finetune/",
        name_prefix="parking_model_ft"
    )

    callbacks = CallbackList([difficulty_callback, checkpoint_callback])

    model.learn(total_timesteps=100_000_000, callback=callbacks)

    model.save("ppo_parking_agent_fine_tuned")
    print("end of training")