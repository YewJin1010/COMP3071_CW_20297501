import pandas as pd
import matplotlib.pyplot as plt

agent = 'A2C-SU'
df = pd.read_csv(f'results/compare_data/agents/{agent}.csv')	

columns = ['Experiment 1', 'Experiment 2', 'Experiment 3', 'Experiment 4']

plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
df[columns].plot(style=['-', '-', '-', '-'], color=['red', 'green', 'blue', 'orange'])
plt.title(f'{agent}')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(f'results/compare_data/agents/plot_{agent}.png')
