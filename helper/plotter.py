import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class Plotter:
    def __init__(self, output_dir="result", file_name="sukp", update_interval=10, interactive=False):
        """
        Visualizer for DQN training metrics on SUKP with online updates.
        
        :param output_dir: str, directory to save plots (e.g., 'result')
        :param file_name: str, base name for plot files (e.g., from SUKPLoader)
        :param update_interval: int, update plots every N episodes
        :param interactive: bool, if True, display plots interactively (non-blocking)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.file_name = file_name
        self.update_interval = update_interval
        self.interactive = interactive
        self.current_episode = 0
        # Initialize lists for metrics
        self.episode_rewards = []
        self.best_profits = []
        self.losses = []
        self.weights = []
        self.entropies = []
        self.terminate_probs = []
        self.capacity = None
        # Set up matplotlib for non-blocking plotting
        if self.interactive:
            plt.ion()  # Enable interactive mode
        plt.switch_backend('Agg')  # Non-blocking backend for saving

    def log_episode(self, total_reward, best_profit, loss, weight, entropy, terminate_prob):
        """
        Logs metrics for one episode and updates plots if interval reached.
        
        :param total_reward: float, sum of rewards in episode
        :param best_profit: float, best profit achieved so far
        :param loss: float, average loss for episode
        :param weight: float, final solution weight
        :param entropy: float, average policy entropy for episode
        :param terminate_prob: float, average terminate probability
        """
        self.episode_rewards.append(total_reward)
        self.best_profits.append(best_profit)
        self.losses.append(loss)
        self.weights.append(weight)
        self.entropies.append(entropy)
        self.terminate_probs.append(terminate_prob)
        self.current_episode += 1
        # Update plots online every update_interval episodes
        if self.current_episode % self.update_interval == 0:
            self.plot_all()

    def set_capacity(self, capacity):
        """
        Sets knapsack capacity for weight plot.
        
        :param capacity: float, knapsack capacity from SetUnionHandler
        """
        self.capacity = capacity

    def _plot_with_moving_avg(self, data, ylabel, title, filename, yline=None):
        """
        Helper to plot data with optional moving average and horizontal line.
        
        :param data: list[float], data to plot
        :param ylabel: str, y-axis label
        :param title: str, plot title
        :param filename: str, output file name (without path)
        :param yline: float, optional horizontal line (e.g., capacity)
        """
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(data) + 1), data, label=ylabel)
        if len(data) >= 50:
            moving_avg = np.convolve(data, np.ones(50)/50, mode='valid')
            plt.plot(range(50, len(data) + 1), moving_avg, label='Moving Avg (50)')
        if yline is not None:
            plt.axhline(y=yline, color='r', linestyle='--', label='Capacity')
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / f"{self.file_name}_{filename}.png")
        if self.interactive:
            plt.draw()
            plt.pause(0.01)  # Non-blocking display
        plt.close()

    def plot_all(self):
        """Generates and saves all diagnostic plots."""
        self._plot_with_moving_avg(self.episode_rewards, 'Total Reward', 'Episode Reward Over Time', 'reward')
        self._plot_with_moving_avg(self.best_profits, 'Best Profit', 'Best Profit Over Time', 'best_profit')
        self._plot_with_moving_avg(self.losses, 'Loss', 'Training Loss Over Time', 'loss')
        if self.capacity is not None:
            self._plot_with_moving_avg(self.weights, 'Total Weight', 'Solution Weight vs. Capacity Over Time', 'weight', yline=self.capacity)
        self._plot_with_moving_avg(self.entropies, 'Average Entropy', 'Policy Entropy Over Time', 'entropy')
        self._plot_with_moving_avg(self.terminate_probs, 'Average Terminate Probability', 'Terminate Probability Over Time', 'terminate_prob')

    def save_metrics(self):
        """Saves logged metrics to a .npy file for later analysis."""
        metrics = {
            'episode_rewards': np.array(self.episode_rewards),
            'best_profits': np.array(self.best_profits),
            'losses': np.array(self.losses),
            'weights': np.array(self.weights),
            'entropies': np.array(self.entropies),
            'terminate_probs': np.array(self.terminate_probs)
        }
        np.save(self.output_dir / f"{self.file_name}_metrics.npy", metrics)