import gymnasium
import copy

class FetchWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.achieved_goal=None
        self.desired_goal=None
        self.success=0
        self.data=None
    
    def reset(self,seed=None):
        super().reset()
        state,info=self.env.reset()

        self.success=0
        self.data=copy.deepcopy(info["data"])

        return state
        
    def step(self, action,sim_state=None):
        
        next_state, reward, terminated,truncated, info = self.env.step(action,sim_state=sim_state)        

        self.data=copy.deepcopy(info["data"])          
        self.success=info["success"]

        # get done
        if terminated or truncated:
            done=True
        else:
            done=False
        
        if self.success:
            reward=10
        else:
            reward=-0.001
        
        return next_state, reward, done, info
