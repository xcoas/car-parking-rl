from parking_env import ParkingEnv
from physics import TEST_BLUEPRINT

env = ParkingEnv(blueprint=TEST_BLUEPRINT, render_mode='human')
obs, info = env.reset()

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("reset")
        obs, info = env.reset()