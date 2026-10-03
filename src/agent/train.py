import sys
import os
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'env')))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env.parking_env import ParkingEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

class DifficultyCallback(BaseCallback):
    def __init__(self, verbose = 1, current_difficulty: float = 0.0):
        super().__init__(verbose)
        self.last_results = deque(maxlen=100)
        self.current_difficulty = current_difficulty

    def _on_step(self) -> bool:
        dones = self.locals['dones']
        infos = self.locals['infos']
        for done, info in zip(dones, infos):
            if done:
                if info['is_success'] == True:
                    self.last_results.append(1)
                else:
                    self.last_results.append(0)

                if (len(self.last_results) == 100):
                    win_percent = sum(self.last_results) / 100
                    if win_percent > 0.7:
                        self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
                        self.training_env.env_method("set_difficulty", self.current_difficulty)
                        print(f"Congrats car just graduated to new difficulty: {self.current_difficulty}, winrate: {win_percent:.1f}")
                        self.last_results.clear()
                        self.model.ep_success_buffer.clear()
                    elif win_percent < 0.2:
                        self.current_difficulty = max(0.0, self.current_difficulty - 0.1)
                        self.training_env.env_method("set_difficulty", self.current_difficulty)
                        print(f"Sadly our car couldnt keep up new difficulty: {self.current_difficulty}, winrate: {win_percent:.1f}")
                        self.last_results.clear()
                        self.model.ep_success_buffer.clear()

        self.logger.record("curriculum/difficulty", self.current_difficulty)
        return True

if __name__ == "__main__":
    test_env = ParkingEnv(initial_difficulty=0.0)
    check_env(test_env)
    print("ParkingEnv passed validation!")

    env = make_vec_env(lambda: ParkingEnv(initial_difficulty=0.0), n_envs=4)

    difficulty_callback = DifficultyCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="parking_model"
    )
    callbacks = CallbackList([difficulty_callback, checkpoint_callback])

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[512, 256]
        )
    )
    model = PPO(
        "MlpPolicy", 
        env, 
        ent_coef=0.00005, 
        verbose=1, 
        policy_kwargs=policy_kwargs,
        n_steps=16000,
        batch_size=4000
    )
    model.learn(total_timesteps=1_000_000_000, callback=callbacks)

    model.save("ppo_parking_agent")
    print("end of training")