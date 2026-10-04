import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_logs.csv")

timesteps = df['timesteps']
difficulty = df['difficulty']
success_rate = df['success_rate']
ep_rew_mean = df['ep_rew_mean']
ep_len_mean = df['ep_len_mean']

plt.figure(figsize=(10, 5))
plt.plot(timesteps, difficulty, color='darkorange', linewidth=2, label='Difficulty level')
plt.plot(timesteps, success_rate, color='green', alpha=0.3, label='Success rate (raw)')
plt.plot(timesteps, success_rate.rolling(window=5, min_periods=1).mean(), color='green', linewidth=2, label='Success rate (moving avg)')
plt.xlabel('Timesteps')
plt.ylabel('Difficulty / Success rate')
plt.title('Curriculum Progress')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('curriculum_progress.png', dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(timesteps, ep_rew_mean, color='royalblue', alpha=0.3, label='Average reward (raw)')
plt.plot(timesteps, ep_rew_mean.rolling(window=5, min_periods=1).mean(), color='darkblue', linewidth=2, label='Average reward (moving avg)')
plt.xlabel('Timesteps')
plt.ylabel('Average reward')
plt.title('Average Episode Reward')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('average_reward.png', dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(timesteps, ep_len_mean, color='darkorange', alpha=0.3, label='Episode steps (raw)')
plt.plot(timesteps, ep_len_mean.rolling(window=5, min_periods=1).mean(), color='orangered', linewidth=2, label='Episode steps (moving avg)')
plt.xlabel('Timesteps')
plt.ylabel('Steps count')
plt.title('Average Episode Steps Count')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('average_steps.png', dpi=150)
plt.close()

print("Charts saved")