import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
import pickle

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
            - mean_rewards : np.ndarray, mean reward per episode across runs
            - std_rewards : np.ndarray, std dev of reward per episode across runs
            - mean_steps : np.ndarray, mean steps per episode across runs
            - std_steps : np.ndarray, std dev of steps per episode across runs
            - runtime_per_run : np.ndarray, duration of each run in seconds
            - total_runtime : float, total duration of the experiment
        """

        rewards = np.zeros((num_episodes, num_runs))
        steps_per_episode = np.zeros((num_episodes, num_runs))
        runtime_per_run = []

        experiment_start = time.time()

        for run in range(num_runs):
            run_start = time.time()
            self.agent.reset()
            run_trajectories = []

            for episode in tqdm(range(num_episodes), desc=f"Run {run+1}/{num_runs} - Episodes", leave=False):
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

                rewards[episode, run] = total_reward
                steps_per_episode[episode, run] = steps

            runtime_per_run.append(time.time() - run_start)

        total_runtime = time.time() - experiment_start

        self.results = {
            "type": "episodic",
            "rewards": rewards,
            "steps": steps_per_episode,
            "mean_rewards": np.mean(rewards, axis=1),
            "std_rewards": np.std(rewards, axis=1),
            "mean_steps": np.mean(steps_per_episode, axis=1),
            "std_steps": np.std(steps_per_episode, axis=1),
            "runtime_per_run": runtime_per_run,
            "total_runtime": total_runtime,
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
            - rewards : np.ndarray, shape (max_episodes, num_runs)
            - steps : np.ndarray, shape (max_episodes, num_runs)
            - runtime_per_run : list of floats
            - total_runtime : float
            - mean_rewards : np.ndarray, mean reward per episode across runs
            - std_rewards : np.ndarray, std reward per episode across runs
            - mean_steps : np.ndarray, mean steps per episode across runs
            - std_steps : np.ndarray, std steps per episode across runs

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
            
        all_run_rewards = []
        all_run_steps = []
        runtime_per_run = []

        total_start_time = time.time()

        for run in range(num_runs):
            run_start = time.time()
            self.agent.reset()
            
            # --- Per-Run Episode Tracking ---
            episode_steps_tracker = np.zeros(num_envs, dtype=int)
            total_rewards_tracker = np.zeros(num_envs, dtype=np.float32)

            run_rewards_collector = []
            run_steps_collector = []
            
            # Reset environment and get initial batch of states (N, state_dim)
            obs, _ = self.env.reset()
            actions = self.agent.start_batch(obs)
                
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

                # 3. Agent Update
                actions = self.agent.step_batch(rewards, next_obs, is_terminal)

                # 4. Process Completed Episodes (Crucial for batch tracking)
                for i in range(num_envs):

                    if is_terminal[i]:
                        # A. Store results
                        run_rewards_collector.append(total_rewards_tracker[i])
                        run_steps_collector.append(episode_steps_tracker[i])

                        # B. Check Quota and Break Cleanly (CRITICAL: Break before reset)
                        episodes_completed_in_run += 1
                        pbar.update(1)
                        
                        if episodes_completed_in_run >= num_episodes:
                            break # Exit inner loop, preventing the next steps from corrupting state
                            
                        # C. Reset episode trackers for this environment
                        total_rewards_tracker[i] = 0.0
                        episode_steps_tracker[i] = 0
                        
                        
                if episodes_completed_in_run >= num_episodes:
                    # Need to break the main while loop as well
                    break 

            if hasattr(self.agent, 'end_batch'):
                self.agent.end_batch(rewards) 
            
            pbar.close()
                
            runtime_per_run.append(time.time() - run_start)
            all_run_rewards.append(run_rewards_collector)
            all_run_steps.append(run_steps_collector)

        total_runtime = time.time() - total_start_time

        # 5. Format Results (Handling variable episode counts per run using np.nan)
        if not all_run_rewards:
            return {}
            
        max_eps = max(len(r) for r in all_run_rewards)
        
        rewards_matrix = np.full((max_eps, num_runs), np.nan)
        steps_matrix = np.full((max_eps, num_runs), np.nan)
        
        for run_idx in range(num_runs):
            r_data = all_run_rewards[run_idx]
            s_data = all_run_steps[run_idx]
            rewards_matrix[:len(r_data), run_idx] = r_data
            steps_matrix[:len(s_data), run_idx] = s_data
            
        self.results = {
            "type": "episodic",
            "rewards": rewards_matrix, 
            "steps": steps_matrix, 
            "runtime_per_run": runtime_per_run,
            "total_runtime": total_runtime,
            "mean_rewards": np.nanmean(rewards_matrix, axis=1),
            "std_rewards": np.nanstd(rewards_matrix, axis=1),
            "mean_steps": np.nanmean(steps_matrix, axis=1),
            "std_steps": np.nanstd(steps_matrix, axis=1),
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
        runtime_data = self.results.get("runtime_per_run", [])
        avg_runtime = np.mean(runtime_data) if len(runtime_data) > 0 else 0
        total_runtime = self.results.get("total_runtime", 0)

        print("="*60)
        print(f" Experiment Summary ({exp_type.capitalize()})")
        print("="*60)
        print(f"Runs:                    {len(runtime_data)}")
        print(f"Total Runtime:           {total_runtime:.3f} seconds")
        print(f"Average Runtime/Run:     {avg_runtime:.3f} seconds")
        print("-"*60)

        if exp_type == "episodic":

            rewards_data = self.results["rewards"]
            steps_data = self.results["steps"]
            num_episodes = rewards_data.shape[0]
            
            # Calculate standard deviations across runs for the summary points
            # Using ddof=1 for sample standard deviation
            def get_stats(data_row):
                return np.nanmean(data_row), np.nanstd(data_row)

            print(f"Episodes per run (Max):  {num_episodes}")
            
            # Reward Statistics
            f_m, f_s = get_stats(rewards_data[0, :])
            l_m, l_s = get_stats(rewards_data[-1, :])
            o_m, o_s = np.nanmean(rewards_data), np.nanstd(rewards_data)
            ln_m, ln_s = np.nanmean(rewards_data[-last_n:]), np.nanstd(rewards_data[-last_n:])

            print(f"\nREWARDS")
            print(f"First Episode:           {f_m:.3f} ± {f_s:.3f}")
            print(f"Last Episode:            {l_m:.3f} ± {l_s:.3f}")
            print(f"Overall Mean:            {o_m:.3f} ± {o_s:.3f}")
            label = f"Last {last_n} Episodes:"
            print(f"{label:<25}{ln_m:.3f} ± {ln_s:.3f}")

            # Steps Statistics
            if steps_data.size > 0:
                sf_m, sf_s = get_stats(steps_data[0, :])
                sl_m, sl_s = get_stats(steps_data[-1, :])
                so_m, so_s = np.nanmean(steps_data), np.nanstd(steps_data)

                print(f"\nSTEPS")
                print(f"First Episode:           {sf_m:.1f} ± {sf_s:.1f}")
                print(f"Last Episode:            {sl_m:.1f} ± {sl_s:.1f}")
                print(f"Overall Mean:            {so_m:.1f} ± {so_s:.1f}")

        elif exp_type == "continuous":

            rewards_data = self.results["rewards"]
            num_steps = rewards_data.shape[0]
            
            o_m, o_s = np.nanmean(rewards_data), np.nanstd(rewards_data)
            ln_m, ln_s = np.nanmean(rewards_data[-last_n:]), np.nanstd(rewards_data[-last_n:])

            print(f"Steps per run:           {num_steps}")
            print(f"\nREWARDS")
            print(f"First Step:              {np.nanmean(rewards_data[0, :]):.3f}")
            print(f"Last Step:               {np.nanmean(rewards_data[-1, :]):.3f}")
            print(f"Overall Mean:            {o_m:.3f} ± {o_s:.3f}")
            print(f"Last {last_n:3d} Steps:         {ln_m:.3f} ± {ln_s:.3f}")

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

    def save_results(self, filepath):
        """
        Saves the experiment results dictionary to a file using pickle.

        Parameters
        ----------
        filepath : str
            Path to the file where results should be saved (e.g., 'results.pkl').
        """
        if not hasattr(self, 'results') or not self.results:
            print("Warning: No results found to save.")
            return

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.results, f)
            print(f"Results successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def load_results(self, filepath):
        """
        Loads experiment results from a pickle file and assigns them to self.results.

        Parameters
        ----------
        filepath : str
            Path to the pickle file.

        Returns
        -------
        dict
            The loaded results dictionary.
        """
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
                
            self.results = loaded_data
            print(f"Results successfully loaded from {filepath}")
            
            # Quick summary of what was loaded
            if isinstance(self.results, dict):
                r_shape = self.results.get('rewards', np.array([])).shape
                print(f"Loaded {self.results.get('type', 'unknown')} results with shape {r_shape}")
                
            return self.results
        except FileNotFoundError:
            print(f"Error: File {filepath} not found.")
        except Exception as e:
            print(f"Error loading results: {e}")
            return None