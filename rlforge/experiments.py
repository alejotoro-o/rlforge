import numpy as np
from tqdm import tqdm
import time

class ExperimentRunner:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.results = {}

    def run_episodic(self, num_runs, num_episodes, max_steps_per_episode):
        rewards = np.zeros((num_episodes, num_runs))
        steps_per_episode = np.zeros((num_episodes, num_runs))
        trajectories = []  # store per-episode trajectories
        runtime_per_run = []

        episodes = np.arange(num_episodes)

        for run in range(num_runs):
            run_start = time.time()
            self.agent.reset()
            run_trajectories = []

            for episode in tqdm(episodes, desc=f"Run {run+1}/{num_runs} - Episodes", leave=False):
                new_state = self.env.reset()[0]
                steps, total_reward, is_terminal = 0, 0, False
                action = self.agent.start(new_state)

                episode_states, episode_actions, episode_rewards = [new_state], [action], []

                while not is_terminal:
                    new_state, reward, terminated, _, _ = self.env.step(action)
                    is_terminal = terminated or steps == max_steps_per_episode - 1
                    action = self.agent.end(reward) if is_terminal else self.agent.step(reward, new_state)

                    total_reward += reward
                    steps += 1

                    episode_states.append(new_state)
                    episode_actions.append(action)
                    episode_rewards.append(reward)

                rewards[episode, run] = total_reward
                steps_per_episode[episode, run] = steps
                run_trajectories.append({
                    "states": episode_states,
                    "actions": episode_actions,
                    "rewards": episode_rewards,
                    "total_reward": total_reward,
                    "steps": steps
                })

            runtime_per_run.append(time.time() - run_start)
            trajectories.append(run_trajectories)

        self.results = {
            "type": "episodic",
            "rewards": rewards,
            "steps": steps_per_episode,
            "trajectories": trajectories,
            "runtime_per_run": runtime_per_run,
            "mean_rewards": np.mean(rewards, axis=1),
            "mean_steps": np.mean(steps_per_episode, axis=1),
        }
        return self.results

    def run_continuous(self, num_runs, num_steps):
        rewards = np.zeros((num_steps, num_runs))
        trajectories = []  # store per-run trajectories
        runtime_per_run = []

        steps = np.arange(num_steps)

        for run in range(num_runs):
            run_start = time.time()
            self.agent.reset()
            new_state = self.env.reset()[0]
            action = self.agent.start(new_state)

            run_states, run_actions, run_rewards = [new_state], [action], []

            for step in tqdm(steps, desc=f"Run {run+1}/{num_runs} - Steps", leave=False):
                new_state, reward, _, _, _ = self.env.step(action)
                action = self.agent.step(reward, new_state)

                rewards[step, run] = reward
                run_states.append(new_state)
                run_actions.append(action)
                run_rewards.append(reward)

            runtime_per_run.append(time.time() - run_start)
            trajectories.append({
                "states": run_states,
                "actions": run_actions,
                "rewards": run_rewards,
                "total_reward": np.sum(run_rewards),
            })

        self.results = {
            "type": "continuous",
            "rewards": rewards,
            "trajectories": trajectories,
            "runtime_per_run": runtime_per_run,
            "mean_rewards": np.mean(rewards, axis=1),
        }
        return self.results

    def summary(self, last_n=10):
        """Print a human-readable summary of the experiment results."""
        if not self.results:
            print("No results available. Run an experiment first.")
            return

        exp_type = self.results.get("type", "unknown")
        avg_runtime = np.mean(self.results.get("runtime_per_run", []))

        print("="*60)
        print(f" Experiment Summary ({exp_type.capitalize()})")
        print("="*60)
        print(f"Runs: {len(self.results.get('runtime_per_run', []))}")
        print(f"Average runtime per run: {avg_runtime:.3f} seconds")

        if exp_type == "episodic":
            num_episodes = self.results["rewards"].shape[0]
            print(f"Episodes per run: {num_episodes}")
            print(f"First episode mean reward: {self.results['mean_rewards'][0]:.3f}")
            print(f"Last episode mean reward: {self.results['mean_rewards'][-1]:.3f}")
            print(f"Overall mean reward: {np.mean(self.results['mean_rewards']):.3f}")
            print(f"Mean reward (last {last_n} episodes): "
                f"{np.mean(self.results['mean_rewards'][-last_n:]):.3f}")

            print(f"First episode mean steps: {self.results['mean_steps'][0]:.1f}")
            print(f"Last episode mean steps: {self.results['mean_steps'][-1]:.1f}")
            print(f"Overall mean steps: {np.mean(self.results['mean_steps']):.1f}")

        elif exp_type == "continuous":
            num_steps = self.results["rewards"].shape[0]
            print(f"Steps per run: {num_steps}")
            print(f"First step mean reward: {self.results['mean_rewards'][0]:.3f}")
            print(f"Last step mean reward: {self.results['mean_rewards'][-1]:.3f}")
            print(f"Overall mean reward: {np.mean(self.results['mean_rewards']):.3f}")
            print(f"Mean reward (last {last_n} steps): "
                f"{np.mean(self.results['mean_rewards'][-last_n:]):.3f}")

        print("="*60)