from tqdm import tqdm
import numpy as np

def runEpisodicExperiment(env, agent, num_runs, num_episodes, max_steps_per_episode):
    """
    Episodic Experiment
    """
    rewards = np.zeros((num_episodes, num_runs))
    steps_per_episode = np.zeros((num_episodes, num_runs))
    episodes = np.arange(num_episodes)

    for run in range(num_runs):

        agent.reset()

        for episode in tqdm(episodes, desc=f"Run {run}/{num_runs} - Episodes", leave=False):

            new_state = env.reset()[0]
            steps = 0
            is_terminal = False
            total_reward = 0

            action = agent.start(new_state)

            while not is_terminal:

                new_state, reward, terminated, _, _ = env.step(action)

                is_terminal = terminated
                if steps == max_steps_per_episode - 1:
                    is_terminal = True

                if is_terminal:
                    action = agent.end(reward)
                else:
                    action = agent.step(reward, new_state)

                total_reward += reward
                steps += 1
            
            rewards[episode, run] = total_reward
            steps_per_episode[episode, run] = steps

    return rewards, steps_per_episode

def runContinuousExperiment(env, agent, num_runs, num_steps):

    rewards = np.zeros((num_steps, num_runs))
    runs = num_runs
    num_steps = np.arange(num_steps)

    for run in range(runs):

        agent.reset()
        new_state = env.reset()[0]  # Reset the environment

        action = agent.start(new_state)

        for step in tqdm(num_steps, desc=f"Run {run}/{runs} - Steps", leave=False):

            new_state, reward, _, _, _ = env.step(action)

            action = agent.step(reward, new_state)

            rewards[step,run] = reward

    return rewards
