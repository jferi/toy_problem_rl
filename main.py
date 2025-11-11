import torch
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

from src.training import train_dqn
from src.evaluation import evaluate_agent, plot_training_metrics

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "DQN MEAN-REVERTING TRADING" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # Hyperparameters
    NUM_EPISODES = 500
    BATCH_SIZE = 128
    MAX_STEPS = 500
    LOOKBACK = 10
    
    print("HYPERPARAMETERS:")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Lookback window: {LOOKBACK}")
    print()
    
    # Train agent
    agent, episode_rewards, losses, epsilons = train_dqn(
        num_episodes=NUM_EPISODES,
        batch_size=BATCH_SIZE,
        max_steps=MAX_STEPS,
        lookback=LOOKBACK,
        print_every=20
    )
    
    # Plot training metrics
    print("\nGenerating learning curves...")
    plot_training_metrics(episode_rewards, losses, epsilons)
    
    # Evaluate agent
    avg_reward, std_reward = evaluate_agent(
        agent,
        num_episodes=10,
        max_steps=MAX_STEPS,
        visualize=True
    )
    
    # Save model
    save_model = input("\nSave model? (y/n): ").lower() == 'y'
    if save_model:
        torch.save({
            'q_network_state_dict': agent.q_network.state_dict(),
            'target_network_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'steps_done': agent.steps_done,
        }, 'dqn_trading_model.pth')
        print("Model saved: dqn_trading_model.pth")
    
    print("\n" + "=" * 80)
    print("PROGRAM COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

