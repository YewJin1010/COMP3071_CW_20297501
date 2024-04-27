# COMP3071_CW_20297501

## Description
This repository contains code for the COMP3071 coursework. It is the GitHub version and the dependencies will not be provided.

## Dependencies
- gym version = 0.25.2
  (Note: Using a version newer than 0.25.2 may cause issues with code execution.)

## How to Use
### Main  
To visualise the experiments, run Main.py.

1. Select the environment:  
For Lunar Lander, enter 1, 2, or 3 for the corresponding experiment:  
    - Experiment 1: Standard Lunar Lander environment  
    - Experiment 2: Modified gravity  
    - Experiment 3: Modified wind  

    No additional input is required for CartPole.  

2. Select an agent by inputting a number from 1 to 5, corresponding to the desired agent.

3.  Enter the number of episodes to run (it is recommended to use 2000.).  

4. Enter the number of experiments to run:  
    - Each experiment is run with the same parameters and results are saved at: 
    ```
    results/{agent}/{experiment}
    ```

### Trainer
To run all agents and experiments automatically without visualization, run Trainer.py.

1. Select the environment:  
    - Enter 1 for Lunar Lander.
    - Enter 2 for CartPole.
2. Enter the number of episodes to run.
3. Enter the number of experiments to run.

**Note**  
For the Trainer, the inputs for gravity and wind are preset to the maximum values of:

    Gravity: -1
    Wind Power: 20
    Wind Turbulence Power: 2
Make sure to follow these instructions when using the code.

## Utils
Additional utilities were utilised for plotting data for report writing:

**plot_experiment**: This utility plots the performance of agents for each experiment.

**plot_agent**: This utility plots the performance of each agent across all experiments.
