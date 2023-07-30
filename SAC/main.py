import sys
import gym
import mujoco_maze  # noqa
from mujoco_maze.maze_env import MazeEnv
sys.path.append("/nfs/home/futuhi/AlphaExploration")
import argparse
from SAC.sac_agent import CustomCallback
from general.maze import Env
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import numpy as np
from stable_baselines3 import SAC
import pickle
import random
from stable_baselines3.common.noise import NormalActionNoise


#hyper params
######################################
replay_buffer_size = 1e6
max_frames  = 6e6
learning_rate=5e-4
max_steps   = 100
batch_size  = 128
seed=random.randint(0,100)
######################################


def plot(address,destinations=None,env=None,success_rates=False):
    
    # plot success rates
    if success_rates is not None:
        pass
    
    
    if destinations is not None:
        # plot coordination of terminal states 
        _, ax = plt.subplots(1, 1, figsize=(5, 4))
        for x, y in env.maze._walls:
            ax.plot(x, y, 'k-')
        
        for sample in destinations:
            
            print(type(sample))
            if sample[0]>9.5 or sample[1]>9.5:
                sample=[min(sample[0],9.5),min(sample[1],9.5)]
            if sample[0]<-0.5 or sample[1]<-0.5:
                sample=[max(sample[0],-0.5),max(sample[1],-0.5)]

            ax.plot(sample[0],sample[1],marker='o',markersize=2,color="red")
        plt.savefig(address+"/destinations.png")

        # record visitation counts
        np.save(address+"/visits",env.density_estimator.visits)

    plt.close("all")



def train(agent,address,environment):
    
    # train and save the model
    agent.learn(max_frames,callback=CustomCallback(address=address,environment=environment))
    agent.save(address+"/sac_agent")



def main(address,environment):
    # initialize the environment and noise
    if environment=="maze":
        env=Env(n=max_steps,maze_type='square_large')
        sigma=0.15 * np.ones(2)
    elif environment=="point":
        env=gym.make("PointUMaze-v1")
        sigma=np.ones(2)* 0.4
        sigma[1]=sigma[1]/8
    else:
        sys.exit("environment is not valid")
    
    # initiate the agent
    action_noise = NormalActionNoise(mean=np.zeros(2), sigma=sigma)
    policy_kwargs = dict(net_arch=dict(pi=[128,128,128], qf=[128,128,128]))
    agent = SAC("MlpPolicy", env=env, verbose=0,buffer_size=int(replay_buffer_size),batch_size=batch_size,policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,seed=seed,device="cuda",train_freq=(12,"step"),action_noise=action_noise)

    # train the agent
    train(agent,address,environment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--address',required=True)
    parser.add_argument('-e','--environment',required=True)
    args = parser.parse_args()
    main(args.address,args.environment)