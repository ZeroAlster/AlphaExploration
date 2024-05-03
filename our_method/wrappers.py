import gym
import numpy as np
from gym import spaces

def get_state(state):
    return np.concatenate((state["observation"],state["desired_goal"]))


class FetchWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.achieved_goal=None
        self.desired_goal=None
    
    def reset(self):
        super().reset()
        state=self.env.reset()
        self.achieved_goal=state["achieved_goal"]
        self.desired_goal=state["desired_goal"]
        return get_state(state)
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.achieved_goal=next_state["achieved_goal"]
        self.desired_goal=next_state["desired_goal"]
        return get_state(next_state), reward, done, info
