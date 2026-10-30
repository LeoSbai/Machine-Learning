import environment
import numpy as np
from agent import DQNAgent
from collections import deque
from agent import ReplayBuffer


class Train:
    """
    Class used for training an agent in an environment.
    """

    def __init__(
        self,
        target_update_steps: int = 1000,
        replay_buffer_size: int = 10_000,
    ):

        # Define the environment and agent
        self.env = environment.CartPole()
        self.agent = DQNAgent(4, 2)

        # Number of steps between each target network update
        self.target_update_steps = target_update_steps

        # Initialise variables
        self.total_steps = 0
        self.memory = ReplayBuffer(replay_buffer_size)

    def _episode(self, replay_sample_size: int = 64, epsilon: float = 1.0):
        """
        Carry out one episode.
        """

        # Reset the environment
        state, _ = self.env.reset()
        finished = False
        ep_steps = 0
        ep_reward = 0

        while not finished:

            # Select an action
            action = self.agent.act(state=state, epsilon=epsilon)

            # Take a step in the environment
            next_state, reward, terminated, truncated, _ = self.env.step(action=action)
            finished = terminated or truncated

            # Save the next_state to the memory
            self.memory.push(state, action, reward, next_state, finished)

            # Once the memory is full, start learning
            if len(self.memory) >= self.memory.max_size():
                self.agent.learn(*self.memory.sample(replay_sample_size))

            # Every target_update_steps, update the target network
            if self.total_steps % self.target_update_steps == 0:
                self.agent.update_target_network()

            # Update
            state = next_state
            ep_steps += 1
            self.total_steps += 1
            ep_reward += reward

            # Decay epsilon
            epsilon = max(0.01, epsilon * 0.99)

        return ep_reward, ep_steps, epsilon

    def train(
        self,
        episodes: int,
        replay_sample_size: int = 64,
        print_results_every: int = 100,
    ):
        """
        Train for a number of episodes and return the avg reward and steps
        """

        # Reset training
        self.total_steps = 0
        epsilon = 1.0

        # History buffers
        reward_history = deque(maxlen=100)
        steps_history = deque(maxlen=100)

        for episode in range(episodes):

            # Train for an episode
            ep_reward, ep_steps, epsilon = self._episode(replay_sample_size, epsilon)

            # Save the episode's results
            reward_history.append(ep_reward)
            steps_history.append(ep_steps)

            # Print avg reward every number of episodes
            if (episode + 1) % print_results_every == 0:
                print(f"Episode {episode+1}: avg reward: {np.mean(reward_history):.2f}")

        return np.mean(reward_history), np.mean(steps_history)
