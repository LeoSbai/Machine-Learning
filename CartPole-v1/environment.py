import gymnasium as gym
import numpy as np


class CartPole(gym.Env):
    """Generates an environment designed to replicate the classic CartPole RL example environment"""

    def __init__(
        self,
        space=10,
        step_limit=None,
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

        self.action_to_force = {0: -force, 1: force}

    def get_obs(self):
        """Returns the state as a dict"""

        return {
            "cart position": self.cart_pos,
            "cart velocity": self.cart_vel,
            "pole position": self.pole_pos,
            "pole velocity": self.pole_vel,
        }

    def get_info(self):
        """Returns auxiliary information for debugging"""

        return {
            "Step": self.steps,
            "Sim time elapsed": self.steps * self.tau,
            "Terminated": self.terminated,
            "Truncated": self.truncated,
        }

    def reset(self, seed=None, options=None):
        """Resets the environment into a random state"""

        super().reset(seed=seed)

        self.terminated = False
        self.truncated = False

        self.steps = 0

        # Cart starts at the center with a random speed
        self.cart_pos = 0
        self.cart_vel = self.np_random.uniform(-0.5, 0.5)

        # Pole starts with a random position but no velocity
        self.pole_pos = self.np_random.uniform(-0.1, 0.1)
        self.pole_vel = 0

        return self.get_obs(), self.get_info()

    def render(self):
        pass

    def dynamics(self, force):
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

        self.steps += 1

        # Find the force
        force = self.action_to_force[action]

        # Implement the force
        cart_acc, pole_acc = self.dynamics(force)

        # Update the state
        self.cart_vel += self.tau * cart_acc
        self.cart_pos += self.tau * self.cart_vel
        self.pole_vel += self.tau * pole_acc
        self.pole_pos += self.tau * self.pole_vel

        # Check for termination
        if abs(self.cart_pos) > self.space:
            reward = -1000
            self.terminated = True
        elif abs(self.pole_pos) > self.termination_angle:
            reward = -100
            self.terminated = True
        else:
            reward = 1

        # Check for truncation
        if (self.step_limit is not None) and (self.steps > self.step_limit):
            self.truncated = True

        observations = self.get_obs()
        info = self.get_info()

        return observations, reward, self.terminated, self.truncated, info
