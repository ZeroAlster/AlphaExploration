import sys
import gym
import gymnasium
import random
import argparse
import numpy as np
import mujoco_maze  # noqa
from mujoco_maze.maze_env import MazeEnv
sys.path.append("/home/futuhi/AlphaExploration")
from wrappers import CustomCallback
from general.maze import Env
from stable_baselines3 import SAC,PPO,TD3,HerReplayBuffer,DDPG
from stable_baselines3.common.noise import NormalActionNoise

#hyper params
######################################
replay_buffer_size = 1e6
max_frames  = 6e6
learning_rate=5e-4
max_steps=100
batch_size  = 128
warmup=20000
######################################


def train(agent,address,environment,method):
    
    # train and save the model
    agent.learn(max_frames,callback=CustomCallback(address=address,environment=environment,method=method))
    agent.save(address+"/model")


def main(address,environment,method,seed):
    # initialize the environment and noise
    if environment=="maze":
        env=Env(n=max_steps,maze_type='square_large',method=method)
        sigma=0.15 * np.ones(2)
        action_noise = NormalActionNoise(mean=np.zeros(2), sigma=sigma)
    elif environment=="point":
        env=gym.make("PointUMaze-v1")
        sigma=np.ones(2)* 0.4
        sigma[1]=sigma[1]/8
        action_noise = NormalActionNoise(mean=np.zeros(2), sigma=sigma)
    elif environment=="push":
        env=gym.make("PointPush-v1")
        sigma=np.ones(2)* 0.4
        sigma[1]=sigma[1]/8
        action_noise = NormalActionNoise(mean=np.zeros(2), sigma=sigma)
    elif environment=="fetch-reach":
        env=gymnasium.make('FetchReach-v2')
        sigma=0.15 * np.ones((1,4))
        action_noise = NormalActionNoise(mean=np.zeros((1,4)), sigma=sigma)
    elif environment=="fetch-push":
        env=gymnasium.make('FetchPush-v2')
        sigma=0.15 * np.ones((1,4))
        action_noise = NormalActionNoise(mean=np.zeros((1,4)), sigma=sigma)
    elif environment=="fetch-slide":
        env=gymnasium.make('FetchSlide-v2')
        sigma=0.15 * np.ones((1,4))
        action_noise = NormalActionNoise(mean=np.zeros((1,4)), sigma=sigma)
    else:
        sys.exit("environment is not valid")
    
    # for SAC and TD3
    goal_selection_strategy = "future"
    policy_kwargs = dict(net_arch=dict(pi=[128,128,128], qf=[128,128,128]))

    if method=="SAC":
        agent = SAC("MultiInputPolicy", env=env, verbose=0,buffer_size=int(replay_buffer_size),batch_size=batch_size,policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,seed=seed,device="cuda",train_freq=(10,"step"),
                action_noise=action_noise,learning_starts=warmup,
                replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=4,goal_selection_strategy=goal_selection_strategy))
    elif method=="PPO":
        agent = PPO("MlpPolicy", env=env, verbose=0,batch_size=batch_size,policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,seed=seed,device="cuda")
    elif method=="TD3":
        agent = TD3("MultiInputPolicy", env=env, verbose=0,buffer_size=int(replay_buffer_size),batch_size=batch_size,policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,seed=seed,device="cuda",train_freq=(12,"step"),action_noise=action_noise,learning_starts=warmup,
                replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=4,goal_selection_strategy=goal_selection_strategy))
    elif method=="DDPG":
        agent = DDPG("MultiInputPolicy", env=env, verbose=0,buffer_size=int(replay_buffer_size),batch_size=batch_size,policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,seed=seed,device="cuda",train_freq=(12,"step"),action_noise=action_noise,learning_starts=warmup)
    else:
        sys.exit("method is not valid")

    # train the agent
    train(agent,address,environment,method)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--address',required=True)
    parser.add_argument('-e','--environment',required=True)
    parser.add_argument('-m','--method',required=True)
    args = parser.parse_args()

    seed=random.randint(0,100)
    main(args.address,args.environment,args.method,seed)