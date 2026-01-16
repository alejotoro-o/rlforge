import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

class ExperimentRunner:
    """
    A unified class to run reinforcement learning experiments.

    This runner supports both episodic and continuous settings across multiple
    runs and environments. It manages agent resets, environment interactions,
    trajectory storage, and provides built-in functionality for summarizing
    and plotting results.

    Parameters
    ----------
    env : object
        The environment instance (standard or vectorized) following the Gym API.
    agent : object
        The agent instance implementing the RL interface (start, step, end, reset).
    """

    def __init__(self, env, agent):
        """
        Initialize the experiment runner.

        Parameters
        ----------
        env : object
            The environment instance.
        agent : object
            The agent instance.
        """

        self.env = env
        self.agent = agent
        self.results = {}

    def _moving_average(self, data, window_size):
        """
        Compute the simple moving average (SMA) of a 1D array.

        Used internally for smoothing plot data. Handles NaN values correctly
        by temporarily treating them as zeros during convolution.

        Parameters
        ----------
        data : np.ndarray
            Input 1D array of values to smooth.
        window_size : int
            Size of the moving average window.

        Returns
        -------
        np.ndarray
            Smoothed array of the same length as input, padded with NaNs
            at the start to maintain alignment.
        """

        if window_size <= 1:
            return data
        # Handle NaN values correctly for the moving average calculation
        weights = np.ones(window_size)
        weights_sum = np.sum(weights)
        
        # Convolve data (treating NaN as 0 temporarily) and weights
        data_nan_to_zero = np.nan_to_num(data)
        smoothed_data = np.convolve(data_nan_to_zero, weights, 'valid') / weights_sum
        
        # Pad the start with NaNs so the length matches the original data length
        # This keeps the x-axis alignment correct for plotting.
        padding = np.full(window_size - 1, np.nan)
        return np.concatenate((padding, smoothed_data))

    def run_episodic(self, num_runs, num_episodes, max_steps_per_episode=None):
        """
        Run the experiment in an episodic setting.

        Executes multiple runs of episodic training, storing rewards,
        steps per episode, and full trajectories.

        Parameters
        ----------
        num_runs : int
            Number of independent runs to execute.
        num_episodes : int
            Number of episodes per run.
        max_steps_per_episode : int, optional
            Maximum steps allowed per episode. If None, episodes run until
            environment termination.

        Returns
        -------
        dict
            Results dictionary containing:
            - type : str, "episodic"
            - rewards : np.ndarray, shape (num_episodes, num_runs)
            - steps : np.ndarray, shape (num_episodes, num_runs)
            - trajectories : list of dicts per run
            - runtime_per_run : list of floats
            - mean_rewards : np.ndarray, mean reward per episode across runs
            - mean_steps : np.ndarray, mean steps per episode across runs
        """

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
                    new_state, reward, terminated, truncated, _ = self.env.step(action)
                    is_terminal = terminated or truncated or (isinstance(max_steps_per_episode, int) and steps == max_steps_per_episode - 1)
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
                    "actions": episode_actions[:-1], # Remove the action after terminal state
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
        """
        Run the experiment in a continuous setting.

        Executes multiple runs of continuous training for a fixed number
        of steps, storing rewards and full trajectories.

        Parameters
        ----------
        num_runs : int
            Number of independent runs to execute.
        num_steps : int
            Number of steps per run.

        Returns
        -------
        dict
            Results dictionary containing:
            - type : str, "continuous"
            - rewards : np.ndarray, shape (num_steps, num_runs)
            - trajectories : list of dicts per run
            - runtime_per_run : list of floats
            - mean_rewards : np.ndarray, mean reward per step across runs
        """

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
                "actions": run_actions[:-1], # Remove the final action taken
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

    def run_episodic_batch(self, num_runs, num_episodes, max_steps_per_episode=None):
        """
        Run the experiment in an episodic setting using a vectorized environment.

        Executes multiple runs of episodic training with parallel environments,
        storing rewards, steps per episode, and full trajectories. This version
        correctly calls the agent's ``end_batch`` method both per environment
        upon episode termination and once at the end of the run for on-policy
        agents (e.g., PPO).

        Parameters
        ----------
        num_runs : int
            Number of independent runs to execute.
        num_episodes : int
            Number of episodes per run.
        max_steps_per_episode : int, optional
            Maximum steps allowed per episode. If None, episodes run until
            environment termination.

        Returns
        -------
        dict
            Results dictionary containing:
            - type : str, "episodic"
            - rewards : np.ndarray, shape (max_episodes, num_runs), padded with NaNs
            - steps : np.ndarray, shape (max_episodes, num_runs), padded with NaNs
            - trajectories : list of lists of dicts per run
            - runtime_per_run : list of floats
            - mean_rewards : np.ndarray, mean reward per episode across runs
            - mean_steps : np.ndarray, mean steps per episode across runs

        Notes
        -----
        - Supports vectorized environments with multiple parallel episodes.
        - Handles per-environment termination and resets trackers correctly.
        """

        
        # Check environment properties
        try:
            num_envs = self.env.num_envs
        except AttributeError:
            num_envs = 1
            
        rewards_list = []
        steps_list = []
        trajectories = [] # store per-run trajectories (list of lists of episode dicts)
        runtime_per_run = []

        for run in range(num_runs):
            run_start = time.time()
            self.agent.reset()
            run_trajectories = []
            
            # --- Per-Run Episode Tracking ---
            episode_steps_tracker = np.zeros(num_envs, dtype=int)
            total_rewards_tracker = np.zeros(num_envs, dtype=np.float32)

            episode_info = [
                {'states': [], 'actions': [], 'rewards': []}
                for _ in range(num_envs)
            ]
            
            # Reset environment and get initial batch of states (N, state_dim)
            obs, _ = self.env.reset()
            actions = self.agent.start_batch(obs)

            # Record initial states and actions for all N parallel trajectories
            for i in range(num_envs):
                episode_info[i]['states'].append(obs[i])
                episode_info[i]['actions'].append(actions[i])
                
            steps_in_run = 0
            episodes_completed_in_run = 0

            pbar = tqdm(total=num_episodes, desc=f"Run {run+1}/{num_runs} - Episodes", leave=False)

            while episodes_completed_in_run < num_episodes:
                
                # 1. Environment Step (N, ...)
                next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
                dones = np.logical_or(terminated, truncated)
                
                # Check for max steps
                is_terminal = dones
                if max_steps_per_episode is not None:
                    max_step_mask = (episode_steps_tracker == max_steps_per_episode - 1)
                    is_terminal = np.logical_or(is_terminal, max_step_mask)

                # 2. Update Trackers (for N parallel environments)
                total_rewards_tracker += rewards
                episode_steps_tracker += 1
                steps_in_run += num_envs

                # 3. Agent Update
                actions = self.agent.step_batch(rewards, next_obs, is_terminal)

                # 4. Process Completed Episodes (Crucial for batch tracking)
                for i in range(num_envs):
                    # Record the current reward, next state, and action taken *next* (A_{t+1})
                    episode_info[i]['rewards'].append(rewards[i])
                    episode_info[i]['states'].append(next_obs[i])
                    episode_info[i]['actions'].append(actions[i])

                    if is_terminal[i]:
                        # A. Store final episode results
                        final_trajectory = {
                            "states": episode_info[i]['states'],
                            # actions list contains [A_0, A_1, ..., A_T, A_{T+1}], we only keep A_0..A_T
                            "actions": np.array(episode_info[i]['actions'])[:-1].tolist(), 
                            "rewards": episode_info[i]['rewards'],
                            "total_reward": total_rewards_tracker[i],
                            "steps": episode_steps_tracker[i] # This is the CORRECT final step count
                        }
                        run_trajectories.append(final_trajectory)
                        rewards_list.append(total_rewards_tracker[i])
                        steps_list.append(episode_steps_tracker[i])

                        # B. Check Quota and Break Cleanly (CRITICAL: Break before reset)
                        episodes_completed_in_run += 1
                        pbar.update(1)
                        
                        if episodes_completed_in_run >= num_episodes:
                            break # Exit inner loop, preventing the next steps from corrupting state
                            
                        # C. Reset episode trackers for this environment (FIX: Reset to 0)
                        total_rewards_tracker[i] = 0.0
                        episode_steps_tracker[i] = 0
                        
                        # D. Prepare for new episode (reset local trajectory buffer)
                        episode_info[i] = {'states': [], 'actions': [], 'rewards': []}
                        
                        # E. Start the new episode's trajectory with the current state (next_obs[i]) and action (actions[i]).
                        episode_info[i]['states'].append(next_obs[i])
                        episode_info[i]['actions'].append(actions[i])
                        
                if episodes_completed_in_run >= num_episodes:
                    # Need to break the main while loop as well
                    break 

            if hasattr(self.agent, 'end_batch'):
                self.agent.end_batch(rewards) 
            
            pbar.close()

            # Final 'end_batch' call for ON-POLICY agents (PPO) only
            # This handles a partial, non-terminal rollout buffer at the end of the *run*.
            # Here, we pass the full N rewards array.
            # if hasattr(self.agent, 'step_count') and self.agent.step_count > 0:
            #     self.agent.end_batch(total_rewards_tracker)
                
            runtime_per_run.append(time.time() - run_start)
            trajectories.append(run_trajectories)

        # 5. Format Results (Handling variable episode counts per run using np.nan)
        if not trajectories:
            self.results = {}
            return self.results
            
        max_episodes_in_any_run = max(len(t) for t in trajectories)
        
        rewards = np.full((max_episodes_in_any_run, num_runs), np.nan)
        steps_per_episode = np.full((max_episodes_in_any_run, num_runs), np.nan)
        
        for run, run_traj in enumerate(trajectories):
            run_rewards = [t["total_reward"] for t in run_traj]
            run_steps = [t["steps"] for t in run_traj]
            rewards[:len(run_rewards), run] = run_rewards
            steps_per_episode[:len(run_steps), run] = run_steps
            
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


    def summary(self, last_n=10):
        """
        Print a summary of experiment results.

        Displays key statistics for episodic or continuous experiments,
        including mean rewards, steps, and runtime information.

        Parameters
        ----------
        last_n : int, optional
            Number of episodes or steps to include in the "last N" summary
            (default=10).

        Notes
        -----
        - Episodic summary includes first, last, overall, and last-N mean rewards
          and steps.
        - Continuous summary includes first, last, overall, and last-N mean rewards.
        - Uses NaN-safe operations to handle padded episodic results.
        - Prints results directly to stdout.

        """

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
            # Using rewards directly as the array might be padded with NaNs
            rewards_data = self.results["rewards"]
            num_episodes = rewards_data.shape[0]
            
            # Use nanmean to correctly handle NaN padding
            print(f"Episodes per run (Max): {num_episodes}")
            print(f"First episode mean reward: {np.nanmean(rewards_data[0, :]):.3f}")
            print(f"Last episode mean reward: {np.nanmean(rewards_data[-1, :]):.3f}")
            print(f"Overall mean reward: {np.nanmean(rewards_data):.3f}")
            print(f"Mean reward (last {last_n} episodes): "
                  f"{np.nanmean(rewards_data[-last_n:]):.3f}")

            # Steps data (optional)
            steps_data = self.results["steps"]
            if steps_data.size > 0:
                print(f"First episode mean steps: {np.nanmean(steps_data[0, :]):.1f}")
                print(f"Last episode mean steps: {np.nanmean(steps_data[-1, :]):.1f}")
                print(f"Overall mean steps: {np.nanmean(steps_data):.1f}")

        elif exp_type == "continuous":
            num_steps = self.results["rewards"].shape[0]
            print(f"Steps per run: {num_steps}")
            print(f"First step mean reward: {self.results['mean_rewards'][0]:.3f}")
            print(f"Last step mean reward: {self.results['mean_rewards'][-1]:.3f}")
            print(f"Overall mean reward: {np.mean(self.results['mean_rewards']):.3f}")
            print(f"Mean reward (last {last_n} steps): "
                  f"{np.mean(self.results['mean_rewards'][-last_n:]):.3f}")

        print("="*60)

    def plot_results(self, metric='reward', window_size=50, max_reward=None):
        """
        Plot experiment results with smoothing and error bands.

        Generates a learning curve or episode length curve depending on the
        selected metric. Results are averaged across runs, smoothed using a
        moving average, and displayed with a shaded error band representing
        the standard deviation.

        Parameters
        ----------
        metric : str, optional
            Metric to plot. Options:
            - "reward" : plots mean total reward across runs.
            - "step"   : plots mean episode length (episodic only).
            Default is "reward".

        window_size : int, optional
            Window size for moving average smoothing (default=50).
        max_reward : float, optional
            Optional maximum reward reference line to plot (default=None).

        Notes
        -----
        - Uses NaN-safe mean and standard deviation calculations to handle
          padded episodic results.
        - Smooths only the mean curve; error bands use raw standard deviation.
        - Supports both episodic and continuous experiment types.
        - Plots include grid, legend, and tight layout for readability.

        Returns
        -------
        None
            Displays a matplotlib plot of the selected metric.
        """

        if not self.results:
            print("No results available. Run an experiment first.")
            return

        if metric == 'reward':
            data = self.results.get('rewards')
            title = "Learning Curve: Mean Total Reward (across Runs)"
            ylabel = "Mean Total Reward"
        elif metric == 'step' and self.results.get('type') == 'episodic':
            data = self.results.get('steps')
            title = "Episode Length Curve: Mean Steps (across Runs)"
            ylabel = "Mean Steps per Episode"
        else:
            print(f"Metric '{metric}' not available for {self.results.get('type')} experiment type.")
            return

        if data is None or data.ndim != 2:
            print(f"Error: Could not retrieve valid 2D data for metric '{metric}'.")
            return

        # 1. Calculate Mean and STD across runs (axis=1) at each time step (episode/step)
        # Use nanmean/nanstd as the arrays may be padded with np.nan if episode lengths differ
        runs_mean = np.nanmean(data, axis=1) # (episodes,) - Raw mean at time t
        runs_std = np.nanstd(data, axis=1)   # (episodes,) - Raw STD at time t

        # 2. Smooth ONLY the Mean using the internal helper
        smoothed_mean = self._moving_average(runs_mean, window_size)
        
        # 3. Use the raw, instantaneous standard deviation for the error band.
        raw_std = runs_std 

        # 4. Prepare for plotting (filter out initial NaN padding)
        n_points = len(smoothed_mean)
        x_axis = np.arange(n_points)

        valid_indices = ~np.isnan(smoothed_mean)
        
        plot_x = x_axis[valid_indices]
        plot_mean = smoothed_mean[valid_indices]
        
        plot_std = raw_std[valid_indices]

        # 5. Calculate the Bounds for the filled area (no arbitrary capping)
        lower_bound = plot_mean - plot_std
        upper_bound = np.minimum(plot_mean + plot_std, np.max(data)) if max_reward is not None else plot_mean + plot_std

        # 6. Plotting
        plt.figure(figsize=(10, 6))

        if max_reward is not None:
            plt.axhline(y=max_reward, color='r', linestyle='--', label=f'Max reward')
        
        # Plot the smoothed mean line
        plt.plot(plot_x, plot_mean, label=f'Smoothed Mean (Window={window_size})', linewidth=2)
        
        # Fill the area for the instantaneous standard deviation (mean ± std)
        plt.fill_between(plot_x, lower_bound, upper_bound, alpha=0.3, label=f'Mean ± STD')
        
        plt.title(f"{title} (Mean Smoothed over {window_size} points)", fontsize=16)
        plt.xlabel(f'{self.results.get("type").capitalize()} Index', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()