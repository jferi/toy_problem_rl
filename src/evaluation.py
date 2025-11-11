import torch
import numpy as np
import matplotlib.pyplot as plt

from .environment import TradingEnvironment
from .agent import DQNAgent
from src.config import EvaluationConfig

def evaluate_agent(
    agent: DQNAgent,
    num_episodes: int = EvaluationConfig.NUM_EPISODES,
    max_steps: int = EvaluationConfig.MAX_STEPS,
    batch_size: int = EvaluationConfig.BATCH_SIZE,
    visualize: bool = EvaluationConfig.VISUALIZE
):
    print("\n" + "=" * 80)
    print("AGENT EVALUATION")
    print("=" * 80)
    
    env = TradingEnvironment(
        batch_size=batch_size,
        lookback=agent.lookback,
        max_steps=max_steps
    )
    
    total_rewards = []
    sample_prices = []
    sample_positions = []
    sample_rewards = []
    
    for episode in range(num_episodes):
        state_price, state_pos = env.reset()
        episode_reward = 0
        
        if episode == 0:
            sample_prices.append(env.current_prices[0].item())
        
        for step in range(max_steps):
            with torch.no_grad():
                actions = agent.select_action(state_price, state_pos, training=False)
            
            (next_state_price, next_state_pos), rewards, dones = env.step(actions)
            episode_reward += rewards[0].item()
            
            if episode == 0:
                sample_prices.append(env.current_prices[0].item())
                sample_positions.append(env.current_position[0].item())
                sample_rewards.append(rewards[0].item())
            
            state_price = next_state_price
            state_pos = next_state_pos
            
            if dones.all():
                break
        
        total_rewards.append(episode_reward)
        print(f"  Episode {episode + 1}: Total reward = {episode_reward:.2f}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"\nAverage total reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    if visualize and len(sample_prices) > 0:
        plot_evaluation(sample_prices, sample_positions, sample_rewards)
    
    return avg_reward, std_reward


def plot_evaluation(prices, positions, rewards):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Price and positions
    ax1 = axes[0]
    time_steps = range(len(prices))
    ax1.plot(time_steps, prices, 'b-', label='Price', alpha=0.7)
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Mean price')
    
    for i in range(len(positions)):
        if positions[i] > 0:  # Long
            ax1.axvspan(i, i+1, alpha=0.3, color='green')
        elif positions[i] < 0:  # Short
            ax1.axvspan(i, i+1, alpha=0.3, color='red')
    
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Price')
    ax1.set_title('Trading Strategy (Green=Long, Red=Short, White=Out)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Positions over time
    ax2 = axes[1]
    ax2.plot(range(len(positions)), positions, 'k-', linewidth=2)
    ax2.fill_between(range(len(positions)), positions, alpha=0.3)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Position')
    ax2.set_title('Position over time')
    ax2.set_ylim([-1.5, 1.5])
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Cumulative reward
    ax3 = axes[2]
    cumulative_rewards = np.cumsum(rewards)
    ax3.plot(range(len(rewards)), cumulative_rewards, 'g-', linewidth=2)
    ax3.fill_between(range(len(rewards)), cumulative_rewards, alpha=0.3, color='green')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Cumulative reward')
    ax3.set_title('Cumulative reward over time')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_training_metrics(episode_rewards, losses, epsilons):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Episode rewards
    ax1 = axes[0]
    ax1.plot(episode_rewards, alpha=0.6, label='Episode reward')
    
    window_size = 20
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(episode_rewards)), moving_avg, 
                'r-', linewidth=2, label=f'{window_size}-episode moving avg')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total reward')
    ax1.set_title('Learning curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    if len(losses) > 0:
        ax2 = axes[1]
        ax2.plot(losses, alpha=0.6)
        ax2.set_xlabel('Training step')
        ax2.set_ylabel('Loss')
        ax2.set_title('TD Loss over time')
        ax2.grid(True, alpha=0.3)
    
    # Epsilon
    ax3 = axes[2]
    ax3.plot(epsilons)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration rate decay')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

