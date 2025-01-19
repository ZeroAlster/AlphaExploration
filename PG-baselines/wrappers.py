import pickle
from stable_baselines3.common.callbacks import BaseCallback
from general.maze import Env
import gym
import gymnasium
import sys
from collections import OrderedDict
import numpy as np
from gymnasium import spaces
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
import imageio

#hyper params
######################################
evaluation_attempts=10
checkpoint=10000
######################################


class GoalEnv(gym.Env):
    """
    Minimal GoalEnv interface compatible with HER.
    """
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute the reward for a given pair of achieved and desired goals.
        Must be overridden by subclasses.
        """
        raise NotImplementedError

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self,address,environment,method,test_env,verbose=0,checkpoint=checkpoint):
        super(CustomCallback, self).__init__(verbose)
        self.checkpoint=checkpoint
        self.locations=[]
        self.success_rates=[]
        self.path=address
        self.environment=environment
        self.test_env=test_env
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
            
            num_success=0
            for _ in range(evaluation_attempts):
                obs,_ = self.test_env.reset()
                done=False
                truncated=False
                success=0
                while (not done) and (not truncated) and (success==0):
                    action,_=self.model.predict(obs, deterministic=True)
                    obs,_, done,truncated,info= self.test_env.step(action)
                    success=info["success"]
                if success!=0:
                    num_success+=1
        
            self.success_rates.append(num_success/evaluation_attempts)

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
    

class HERWrapper(gymnasium.Wrapper, GoalEnv):
    def __init__(self, env):        
        super(HERWrapper, self).__init__(env)
        self.env = env
        
        # Assume observation space includes goal; modify as needed
        obs_space = self.env.observation_space.shape[0]  # Adjust based on your env
        goal_space = 2  # Assume goal is 2-dimensional

        # Define the observation space as a Dict
        self.observation_space = spaces.Dict(OrderedDict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(goal_space,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(goal_space,), dtype=np.float32),
        }))
        self.action_space = self.env.action_space

        # print("state space: "+str(self.observation_space))
        # print("actions space: "+str(self.action_space))

    def reset(self,seed):
        obs = self.env.reset()
        return self._split_observation(obs)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        split_obs = self._split_observation(next_obs)
        return split_obs, reward, done, info

    def _split_observation(self, obs):
        """
        Split the observation into observation, achieved_goal, and desired_goal.
        Modify this function based on your observation structure.
        """
        # Assuming the last two entries of `obs` are goals (adjust as necessary)
        observation = obs[:2]
        achieved_goal = obs[:2]  # Example: last two values are the achieved goal
        desired_goal = obs[-2:]  # Example: last two values are the desired goal (modify as needed)
        
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute the reward based on the distance between achieved_goal and desired_goal.
        Modify as needed for your task.
        """
        # Example reward: negative L2 distance
        return -np.linalg.norm(achieved_goal - desired_goal)




class MujocoWrapper(gym.Wrapper, GoalEnv):
    def __init__(self, env):        
        super(MujocoWrapper, self).__init__(env)
        self.env = env
        
        # Assume observ`ation space includes goal; modify as needed
        obs_space = self.env.observation_space.shape[0]  # Adjust based on your env
        goal_space = 2  # Assume goal is 2-dimensional

        # Define the observation space as a Dict
        self.observation_space = spaces.Dict(OrderedDict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(goal_space,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(goal_space,), dtype=np.float32),
        }))
        self.action_space = self.env.action_space

        # print("state space: "+str(self.observation_space))
        # print("actions space: "+str(self.action_space))

    def reset(self,seed):
        obs = self.env.reset()
        return self._split_observation(obs)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        split_obs = self._split_observation(next_obs)
        return split_obs, reward, done, info

    def _split_observation(self, obs):
        """
        Split the observation into observation, achieved_goal, and desired_goal.
        Modify this function based on your observation structure.
        """
        # Assuming the last two entries of `obs` are goals (adjust as necessary)
        observation = obs[:2]
        achieved_goal = obs[:2]  # Example: last two values are the achieved goal
        desired_goal = obs[-2:]  # Example: last two values are the desired goal (modify as needed)
        
        
        print("obs:"+str(obs))
        print("observation:"+str(observation))
        print("achieved goal:"+str(achieved_goal))
        print("desired goal:"+str(desired_goal))
        sys.exit()
        
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute the reward based on the distance between achieved_goal and desired_goal.
        Modify as needed for your task.
        """
        # Example reward: negative L2 distance
        return -np.linalg.norm(achieved_goal - desired_goal)
