import argparse
from env import Env
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import numpy as np
from agent import CustomCallback
from stable_baselines3 import SAC
import pickle
import random


#hyper params
######################################
replay_buffer_size = 1e6
max_frames  = 5e6
learning_rate=0.0006
max_steps   = 50
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



def train(agent,env,address):
    
    # train and save the model
    agent.learn(max_frames,callback=CustomCallback(address=address))
    agent.save(address+"/sac_agent")

    # plot reached states at the end of each episode
    destinations=[]
    with open(address+"/locations", 'rb') as fp:
        destinations = pickle.load(fp)
    plot(address,destinations=destinations,env=env)



def main(address):
    # initialize the environment
    env=Env(n=max_steps,maze_type='square_large')
    
    # initiate the agent
    policy_kwargs = dict(net_arch=dict(pi=[128,128,128], qf=[128,128,128]))
    agent = SAC("MlpPolicy", env=env, verbose=0,buffer_size=int(replay_buffer_size),batch_size=batch_size,policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,seed=seed)

    # train the agent
    train(agent,env,address)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--address',required=True)
    args = parser.parse_args()
    main(args.address)