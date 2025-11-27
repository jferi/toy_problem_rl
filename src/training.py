import torch
import numpy as np

from src.config import Config
from .environment import TradingEnvironment
from .agent import DQNAgent


def train_dqn(conf: Config):
    print("=" * 80)
    print("DQN TRAINING")
    print("=" * 80)
    
    env = TradingEnvironment(conf)
    agent = DQNAgent(conf)
    
    episode_rewards = []
    episode_lengths = []
    losses = []
    epsilons = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for episode in range(conf.num_episodes):
        state_price, state_pos = env.reset()
        episode_reward = torch.zeros(conf.batch_size, device=device)
        
        for step in range(conf.max_steps):
            actions = agent.select_action(state_price, state_pos, training=True)
            (next_state_price, next_state_pos), rewards, dones = env.step(actions)
            
            episode_reward += rewards
            
            agent.memory.push(
                state_price, state_pos, actions, rewards,
                next_state_price, next_state_pos, dones
            )
            
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            
            state_price = next_state_price
            state_pos = next_state_pos
            
            if dones.all():
                break
        
        mean_reward = episode_reward.mean().item()
        episode_rewards.append(mean_reward)
        episode_lengths.append(step + 1)
        epsilons.append(agent.epsilon)
        
        if (episode + 1) % conf.print_every == 0:
            avg_reward = np.mean(episode_rewards[-conf.print_every:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            
            print(f"\nEpisode {episode + 1}/{conf.num_episodes}")
            print(f"  Avg reward (last {conf.print_every} ep): {avg_reward:.3f}")
            print(f"  Avg loss (last 100 steps): {avg_loss:.6f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Replay buffer size: {len(agent.memory)}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    return agent, episode_rewards, losses, epsilons
