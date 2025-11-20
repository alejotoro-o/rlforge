import numpy as np
from tqdm import tqdm
import time

class ExperimentRunner:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.results = {}

    def run_episodic(self, num_runs, num_episodes, max_steps_per_episode=None):
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

    def run_episodic_batch(self, num_runs, num_episodes, max_steps_per_episode=None):
        """
        Runs the experiment using a vectorized environment and the agent's batch methods.
        """
        
        # Check environment properties
        try:
            num_envs = self.env.num_envs
        except AttributeError:
            num_envs = 1
            
        rewards_list = []
        steps_list = []
        trajectories = []  # store per-run trajectories (list of lists of episode dicts)
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

            # --- Main Loop ---
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
            
            pbar.close()

            # Handle agent's final update (if any) if the loop ended mid-rollout
            # This is CORRECTLY placed outside the main loop.
            if self.agent.step_count > 0:
                self.agent.end_batch(total_rewards_tracker)
                
            runtime_per_run.append(time.time() - run_start)
            trajectories.append(run_trajectories)

        # 5. Format Results (This section is fine)
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
            "mean_rewards": np.nanmean(rewards, axis=1),
            "mean_steps": np.nanmean(steps_per_episode, axis=1),
        }
        return self.results