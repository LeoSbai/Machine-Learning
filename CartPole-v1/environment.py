import gymnasium as gym
import numpy as np


class CartPole(gym.Env):
    """Generates an environment designed to replicate the classic CartPole RL example environment"""

    def __init__(
        self,
        space=10,
        step_limit=10_000,
        termination_angle=np.pi / 2,
        acceleration=1,
        step_duration=0.1,
    ):

        self.space = space  # m
        self.step_duration = step_duration  # s

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

        self.observation_space = gym.spaces.Dict(
            {
                "cart position": gym.spaces.Box(-space, space),
                "cart velocity": gym.spaces.Box(-np.inf, np.inf),
                "pole position": gym.spaces.Box(
                    -self.termination_angle, self.termination_angle
                ),
                "pole velocity": gym.spaces.Box(-np.inf, np.inf),
            }
        )

        # Two actions, left and right force
        self.action_space = gym.spaces.Discrete(2)

        self.action_to_direction = {0: -acceleration, 1: acceleration}

    def get_obs(self):
        """Returns the state as a dict"""

        return {
            "Cart position": self.cart_pos,
            "Cart velocity": self.cart_vel,
            "Pole angle": self.pole_pos,
            "Pole angular velocity": self.pole_vel,
        }

    def get_info(self):
        """Returns auxiliary information for debugging"""

        return {
            "Step": self.steps,
            "Sim time elapsed": self.steps * self.step_duration,
            "Terminated": self.terminated,
            "Truncated": self.truncated,
        }

    def reset(self, seed=None, options=None):
        """Resets the environment into a random state"""

        super().reset(seed=seed)

        self.steps = 0

        # Cart starts at the center with a random speed
        self.cart_pos = 0
        self.cart_vel = self.np_random.uniform(-0.5, 0.5)

        # Pole starts with a random position but no velocity
        self.pole_pos = self.np_random.uniform(-0.1, 0.1)
        self.pole_vel = 0

    def render():
        pass

    def step(self, action):
        """Carries out one step after an action has been taken"""
