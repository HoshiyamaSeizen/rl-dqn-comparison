import os
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import trange

from DQN import DQNAgent
from DDQN import DDQNAgent
from PrioritizedReplay import PrioritizedReplayAgent
from DuelingDQN import DuelingDQNAgent

os.makedirs("plots", exist_ok=True)

agents = [DQNAgent, DDQNAgent, PrioritizedReplayAgent, DuelingDQNAgent]
environments = [
    {"name": "CartPole-v1", "state_dim": 4, "action_dim": 2}, 
    {"name": "LunarLander-v3", "state_dim": 8, "action_dim": 4}, 
    {"name": "MountainCar-v0", "state_dim": 2, "action_dim": 3}
]


params = {
    "gamma": 0.99,
    "epsilon": 0.7,
    "epsilon_decay": 0.955,
    "epsilon_min": 0.05,
    "layers": [128, 128],
    "num_steps": 200,
    "num_episodes": 300,
}

def shaped_reward(state, action, original_reward, next_state):
    position = state[0]
    new_position = next_state[0]
    new_velocity = next_state[1]

    position_reward = 10 * (new_position - position) # Награда за движение в сторону гола
    velocity_reward = 0.1 * abs(new_velocity) # Награда за высокую скорость
    action_penalty = -0.01 if action != 1 else 0 # Пенальти за отсутсвие действий

    return position_reward + velocity_reward + action_penalty

def train(agent, env, episodes, steps, reshape_reward=False):
    reward_history = []
    loss_history = []

    for ep in trange(episodes, desc="Эпизоды"):
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0

        for _ in range(steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            reward = shaped_reward(state, action, reward, next_state) if reshape_reward else reward
            agent.buffer.push(state, action, reward, next_state, float(done))
            loss = agent.train_step()
            state = next_state
            total_reward += reward
            total_loss += loss
            if done or truncated:
                break
        reward_history.append(total_reward)
        loss_history.append(total_loss)
        agent.update_epsilon()
        agent.update_target()
    return reward_history, loss_history

def plot_results(results, title, filename):
    for metric, idx in [('reward', 0), ('loss', 1)]:
        plt.figure(figsize=(10, 6))
        for label, data in results.items():
            plt.plot(data[idx], label=label)
        plt.title(f"{title}")
        plt.xlabel("Iteration")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/experiment_{filename}_{metric}.png")
        plt.close()

def experiment(environment):
    env = gym.make(environment["name"])
    results = {}
    for agentClass in agents:
        agent = agentClass(environment["state_dim"], environment["action_dim"], params["layers"], params["gamma"], params["epsilon"], params["epsilon_decay"])
        rewards, losses = train(agent, env, params["num_episodes"], params["num_steps"], environment["name"]=="MountainCar-v0")
        results[type(agent).__name__[:-5]] = (rewards, losses)
    return results

if __name__ == "__main__":
    print("Running CartPole experiment...")
    r_CartPole = experiment(environments[0])
    plot_results(r_CartPole, "Experiment", "CartPole")

    print("Running LunarLander experiment...")
    r_LunarLander = experiment(environments[1])
    plot_results(r_LunarLander, "Experiment", "LunarLander")

    print("Running MountainCar experiment...")
    r_MountainCar = experiment(environments[2])
    plot_results(r_MountainCar, "Experiment", "MountainCar")