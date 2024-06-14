import sys
import gym
import gymnasium
import random
import argparse
import numpy as np
import mujoco_maze  # noqa
from mujoco_maze.maze_env import MazeEnv
sys.path.append("/home/futuhi/AlphaExploration")
from wrappers import CustomCallback,RewardWrapper
from general.maze import Env
from stable_baselines3 import SAC,TD3,DDPG
from stable_baselines3.common.noise import NormalActionNoise
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
import os

#hyper params
######################################
replay_buffer_size = 1e6
max_frames  = 2e6
learning_rate=3e-4
max_steps=100
batch_size  = 512
warmup=20000
######################################


def train(agent,address,environment,method,test_env):
    
    # train and save the model
    agent.learn(max_frames,callback=CustomCallback(test_env=test_env,address=address,environment=environment,method=method))
    agent.save(address+"/model")


def main(address,environment,method,seed):
    # initialize the environment and noise
    if "v2" not in environment:
        if environment=="maze":
            env=Env(n=max_steps,maze_type='square_large',method=method)
            test_env=Env(n=max_steps,maze_type='square_large',method=method)
            sigma=0.15 * np.ones(2)
            action_noise = NormalActionNoise(mean=np.zeros(2), sigma=sigma)
        elif environment=="point":
            env=gym.make("PointUMaze-v1")
            test_env=gym.make("PointUMaze-v1")
            sigma=np.ones(2)* 0.4
            sigma[1]=sigma[1]/8
            action_noise = NormalActionNoise(mean=np.zeros(2), sigma=sigma)
        elif environment=="push":
            env=gym.make("PointPush-v1")
            test_env=gym.make("PointPush-v1")
            sigma=np.ones(2)* 0.4
            sigma[1]=sigma[1]/8
            action_noise = NormalActionNoise(mean=np.zeros(2), sigma=sigma)
        else:
            raise ValueError("The environment does not exist.")
    else:
        env=RewardWrapper(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[environment+"-goal-observable"](render_mode="rgb_array",seed=seed))
        test_env=RewardWrapper(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[environment+"-goal-observable"](render_mode="rgb_array",seed=seed))
        sigma=0.15 * np.ones((1,4))
        action_noise = NormalActionNoise(mean=np.zeros((1,4)), sigma=sigma)
    
    # for SAC,TD3, and DDPG 
    policy_kwargs = dict(net_arch=dict(pi=[128,128,128], qf=[128,128,128]))

    print('Setting environment variable to use GPU rendering:')
    os.environ["MUJOCO_GL"]="egl"

    if method=="SAC":
        agent = SAC("MlpPolicy", env=env, verbose=0,buffer_size=int(replay_buffer_size),batch_size=batch_size,policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,seed=seed,device="cuda",train_freq=(2,"step"),action_noise=action_noise)
    elif method=="TD3":
        agent = TD3("MlpPolicy", env=env, verbose=0,buffer_size=int(replay_buffer_size),batch_size=batch_size,policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,seed=seed,device="cuda",train_freq=(2,"step"),action_noise=action_noise)
    elif method=="DDPG":
        agent = DDPG("MlpPolicy", env=env, verbose=0,buffer_size=int(replay_buffer_size),batch_size=batch_size,policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,seed=seed,device="cuda",train_freq=(2,"step"),action_noise=action_noise)
    else:
        raise ValueError("method is not valid")

    # train the agent
    train(agent,address,environment,method,test_env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--address',required=True)
    parser.add_argument('-e','--environment',required=True)
    parser.add_argument('-m','--method',required=True)
    args = parser.parse_args()

    seed=random.randint(0,100)
    main(args.address,args.environment,args.method,seed)