import pickle
from stable_baselines3.common.callbacks import BaseCallback
from general.maze import Env
import math
import gym
import sys

#hyper params
######################################
evaluation_attempts=10
######################################


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self,address,environment, verbose=0,checkpoint=10000):
        super(CustomCallback, self).__init__(verbose)
        self.checkpoint=checkpoint
        self.locations=[]
        self.success_rates=[]
        self.path=address
        self.environment=environment
        self.success=0
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

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:

        if self.locals["dones"][0]:
            if math.sqrt(math.pow(self.locals["infos"][0]["terminal_observation"][0]-self.goal[0],2)+
                         math.pow(self.locals["infos"][0]["terminal_observation"][1]-self.goal[1],2))<=self.threshold:
                self.success+=1
            self.locations.append(self.locals["infos"][0]["terminal_observation"])

        
        if self.num_timesteps % self.checkpoint==0:
            print("next checkpoint: "+str(self.num_timesteps)+"  steps")
            print("goal is achieved: "+str(self.success))
            
            if self.environment=="point":
                env = gym.make("PointUMaze-v1")
            elif self.environment=="maze":
                env=Env(n=self.len_episode,maze_type='square_large')
            elif self.environment=="push":
                env=gym.make("PointPush-v1")

            success=0
            for _ in range(evaluation_attempts):
                obs = env.reset()
                done=False
                while not done:
                    action, _states=self.model.predict(obs, deterministic=False)
                    obs,r, done,_= env.step(action)
                if math.sqrt(math.pow(obs[0]-self.goal[0],2)+
                         math.pow(obs[1]-self.goal[1],2))<=self.threshold:
                    success+=1
            self.success_rates.append(success/evaluation_attempts)

        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        with open(self.path+"/locations", "wb") as fp:
            pickle.dump(self.locations, fp)

        with open(self.path+"/success_rates", "wb") as fp:
            pickle.dump(self.success_rates, fp)