import gymnasium as gym
import numpy as np


class CartPole(gym.Env):
    """An environment designed to replicate the classic CartPole RL example environment"""

    def __init__(self, space=10, step_limit=10_000):

        self.space = space

        self.step_limit = step_limit
        self.steps = None

        # End conditions
        self.terminated = False
        self.truncated = False

        # Cart features
        self.cart_pos = None
        self.cart_vel = None

        # Pole features
        self.pol_pos = None
        self.pol_vel = None

    def get_obs():
        pass

    def get_info():
        pass

    def reset():
        pass

    def render():
        pass

    def step():
        pass
