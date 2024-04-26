import pandas as pd
import matplotlib.pyplot as plt

experiment = 'experiment4'
df = pd.read_csv(f'results/compare_data/experiments/{experiment}.csv')	

columns = ['DQN', 'PPO', 'A2C', 'A2C-MLP','A2C-SU']

plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
df[columns].plot(style=['-', '-', '-', '-', '-'], color=['red', 'green', 'blue', 'orange', 'purple'])
plt.title('Experiment 4: Lunar Lander Random Wind and Turbulence')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(f'results/compare_data/experiments/plot_{experiment}.png')
