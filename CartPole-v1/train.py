import environment
import numpy as np
from agent import DQNAgent
from collections import deque


class Train:
    """
    Class used for training an agent in an environment.
    """

    def __init__(self, target_update_steps):

        self.env = environment.CartPole()
        self.agent = DQNAgent(4, 2)

        # Number of steps between each target network update
        self.target_update_steps = target_update_steps

        # Total steps for target network updating
        self.total_steps = 0

    def _episode(self):
        """
        Carry out one episode.
        """
        # Reset the environment
        obs, _ = self.env.reset()
        finished = False
        ep_steps = 0
        ep_reward = 0

        while not finished:
            action = self.agent.act(state=obs)
            new_obs, reward, terminated, truncated, _ = self.env.step(action=action)
            finished = terminated or truncated
            self.agent.learn(obs, action, reward, new_obs, finished)

            obs = new_obs

            ep_steps += 1
            self.total_steps += 1
            ep_reward += reward

            if self.total_steps % self.target_update_steps == 0:
                self.agent.update_target_network()

        return ep_reward, ep_steps

    def train(self, episodes):
        """
        Train for a number of episodes and return the avg reward and steps
        """

        self.total_steps = 0

        reward_history = deque(maxlen=100)
        steps_history = deque(maxlen=100)

        # Train for a number of episodes and log results
        for episode in range(episodes):

            ep_reward, ep_steps = self._episode()

            reward_history.append(ep_reward)
            steps_history.append(ep_steps)

            # Print avg results every 100 episodes
            if episode + 1 % 100 == 0:
                print(
                    f"Episode {episode+1}: avg reward: {np.mean(reward_history):.2f} | avg steps: {np.mean(steps_history):.2f}"
                )

        return np.mean(reward_history), np.mean(steps_history)
