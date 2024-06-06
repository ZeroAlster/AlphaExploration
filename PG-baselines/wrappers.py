import pickle
from stable_baselines3.common.callbacks import BaseCallback
from general.maze import Env
import gym
import gymnasium
import sys
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
import imageio

#hyper params
######################################
evaluation_attempts=10
checkpoint=10000
######################################


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self,address,environment,method,verbose=0,checkpoint=checkpoint):
        super(CustomCallback, self).__init__(verbose)
        self.checkpoint=checkpoint
        self.locations=[]
        self.success_rates=[]
        self.path=address
        self.environment=environment
        self.success=0
        self.method=method
        if self.environment=="point":
            self.len_episode=500
            self.goal=[0,16]
            self.threshold=0.6
        elif self.environment=="maze":
            self.len_episode=100
            self.goal=[8.8503,9.1610]
            self.threshold=0.15
        elif self.environment=="push":
            self.len_episode=500
            self.goal=[4,24.8]
            self.threshold=0.6
        else:
            self.threshold=None

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        
        if self.locals["infos"][0]["success"]!=0:
            self.success+=1
        

        if self.num_timesteps % self.checkpoint==0:
            print("next checkpoint: "+str(self.num_timesteps)+"  steps")
            print("goal is achieved: "+str(self.success))
            
            if self.environment=="point":
                env = gym.make("PointUMaze-v1")
            elif self.environment=="maze":
                env=Env(n=self.len_episode,maze_type='square_large',method=self.method)
            elif self.environment=="push":
                env=gym.make("PointPush-v1")
            else:
                env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[self.environment+"-goal-observable"](render_mode="rgb_array")

            success=0
            for _ in range(evaluation_attempts):
                obs,_ = env.reset()
                done=False
                truncated=False
                while (not done) and (not truncated):
                    action,_=self.model.predict(obs, deterministic=True)
                    obs,_, done,truncated,info= env.step(action)
                if info["success"]!=0:
                    success+=1
        
            self.success_rates.append(success/evaluation_attempts)

        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        # with open(self.path+"/locations", "wb") as fp:
        #     pickle.dump(self.locations, fp)

        with open(self.path+"/success_rates", "wb") as fp:
            pickle.dump(self.success_rates, fp)


class RewardWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action,sim_state=None):
        
        next_state, reward, terminated,truncated, info = self.env.step(action,sim_state=sim_state)        
        
        if info["success"]:
            reward=10
        else:
            reward=-0.001
        
        return next_state, reward,terminated,truncated, info
