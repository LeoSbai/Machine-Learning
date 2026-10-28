import gymnasium as gym
import numpy as np


class CartPole(gym.Env):
    """Generates an environment designed to replicate the classic CartPole RL example environment"""

    def __init__(self, space=10, step_limit=10_000, termination_angle=np.pi / 2):

        self.space = space  # m

        self.step_limit = step_limit
        self.steps = None

        # End conditions
        self.terminated = False
        self.truncated = False

        # Cart features
        self.cart_pos = None  # m
        self.cart_vel = None  # m/s

        # Pole features
        self.pole_pos = None  # rad
        self.pole_vel = None  # rad/s
        self.termination_angle = termination_angle

    def get_obs(self):
        """Returns the state as a dict"""

        return {
            "Cart position": self.cart_pos,
            "Cart velocity": self.cart_vel,
            "Pole angle": self.pole_pos,
            "Pole angular velocity": self.pol_vel,
        }

    def get_info(self):
        """Returns auxiliary information for debugging"""

        return {
            "Step": self.steps,
            "Terminated": self.terminated,
            "Truncated": self.truncated,
        }

    def reset(self, seed=None, options=None):
        """Resets the environment into a random state"""

        super().reset(seed=seed)

        # Cart starts at the center with a random speed
        self.cart_pos = 0
        self.cart_vel = self.np_random.uniform(-0.5, 0.5)

        # Pole starts with a random position but no velocity
        self.pole_pos = self.np_random.uniform(-0.1, 0.1)
        self.pole_vel = 0

    def render():
        pass

    def step():
        pass
