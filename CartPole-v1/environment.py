import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class CartPole(gym.Env):
    """Generates an environment designed to replicate the classic CartPole RL example environment"""

    def __init__(
        self,
        space=10,
        step_limit=500,
        termination_angle=np.pi / 2,
        force=10,
        tau=0.01,  # Step duration
        m_cart=10,
        m_pole=1,
        pole_length=1,
        gravity=9.81,
    ):

        self.space = space  # m
        self.tau = tau  # s
        self.m_cart = m_cart  # kg
        self.m_pole = m_pole  # kg
        self.pole_length = pole_length  # m
        self.gravity = gravity  # m/s^2

        self.step_limit = step_limit
        self.steps = 0

        # Cart features
        self.cart_pos = 0.0  # m
        self.cart_vel = 0.0  # m/s

        # Pole features
        self.pole_pos = 0.0  # rad
        self.pole_vel = 0.0  # rad/s
        self.termination_angle = termination_angle

        # Observation space twice as wide as limits to prevent clipping at terminal state
        self.observation_space = gym.spaces.Box(
            low=np.array(
                [-space * 2, -np.inf, -termination_angle * 2, -np.inf], dtype=np.float32
            ),
            high=np.array(
                [space * 2, np.inf, termination_angle * 2, np.inf], dtype=np.float32
            ),
            dtype=np.float32,
        )

        # Two actions, left and right force
        self.action_space = gym.spaces.Discrete(2)

        self.action_to_force = {0: -force, 1: force}

    def _get_obs(self):
        """Returns the state as a dict"""

        return np.array(
            [self.cart_pos, self.cart_vel, self.pole_pos, self.pole_vel],
            dtype=np.float32,
        )

    def _get_info(self):
        """Returns auxiliary information for debugging"""

        return {
            "Step": self.steps,
            "Sim time elapsed": self.steps * self.tau,
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

        return self._get_obs(), self._get_info()

    def render(self):

        if not hasattr(self, "fig"):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 3))

        self.ax.clear()

        # Track
        self.ax.plot([-self.space, self.space], [0, 0], "k-", lw=2)

        # Cart
        cart_w = 0.5
        cart_h = 0.25

        self.ax.add_patch(
            Rectangle(
                (self.cart_pos - cart_w / 2, -cart_h / 2),
                cart_w,
                cart_h,
                color="blue",
            )
        )

        # Pole
        pole_x = self.cart_pos + self.pole_length * np.sin(self.pole_pos)
        pole_y = self.pole_length * np.cos(self.pole_pos)

        self.ax.plot(
            [self.cart_pos, pole_x],
            [0, pole_y],
            "r-",
            lw=4,
        )

        self.ax.set_xlim(-self.space, self.space)
        self.ax.set_ylim(-0.5, self.pole_length + 0.5)
        self.ax.set_aspect("equal")

        plt.pause(0.001)

    def _dynamics(self, force):
        costheta = np.cos(self.pole_pos)
        sintheta = np.sin(self.pole_pos)

        temp = (
            force + (self.m_pole * self.pole_length / 2) * self.pole_vel**2 * sintheta
        ) / (self.m_pole + self.m_cart)

        pole_acc = (self.gravity * sintheta - costheta * temp) / (
            self.pole_length
            / 2
            * (4.0 / 3.0 - self.m_pole * costheta**2 / (self.m_pole + self.m_cart))
        )

        cart_acc = temp - (
            (self.m_pole * self.pole_length / 2) * pole_acc * costheta
        ) / (self.m_pole + self.m_cart)

        return cart_acc, pole_acc

    def step(self, action):
        """Carries out one step after an action has been taken"""

        # Check for invalid action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1

        # Find the force
        force = self.action_to_force[action]

        # Implement the force
        cart_acc, pole_acc = self._dynamics(force)

        # Update the state
        self.cart_vel += self.tau * cart_acc
        self.cart_pos += self.tau * self.cart_vel
        self.pole_vel += self.tau * pole_acc
        self.pole_pos += self.tau * self.pole_vel

        # Check for terminal conditions
        terminated = (
            abs(self.cart_pos) > self.space
            or abs(self.pole_pos) > self.termination_angle
        )
        truncated = (self.step_limit is not None) and (self.steps >= self.step_limit)

        reward = 1

        observations = self._get_obs()
        info = self._get_info()

        return observations, reward, terminated, truncated, info
