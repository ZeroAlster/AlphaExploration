import gymnasium
import numpy as np
import sys

class FetchWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.achieved_goal=None
        self.desired_goal=None
        self.success=0
    
    def reset(self,seed=None):
        super().reset()
        state,_=self.env.reset()

        self.success=0

        return state
        
    def step(self, action,sim_state=None):
        
        next_state, reward, terminated,truncated, info = self.env.step(action,sim_state=sim_state)

        # get success
        self.success=info["success"]

        # get done
        if terminated or truncated:
            done=True
        else:
            done=False
        
        return next_state, reward, done, info
