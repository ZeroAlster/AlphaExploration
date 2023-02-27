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
import statistics



#hyper params
######################################
replay_buffer_size = 1e6
max_frames  = 5e6
learning_rate=0.0006
max_steps   = 50
batch_size  = 128
seed=random.randint(0,100)
num_agents=5
checkpoints_interval=10000
######################################


def plot(address,destinations=None,env=None,success_rates=None):
    
    # plot success rates
    if success_rates is not None:
        std=np.zeros((1,len(success_rates[0])))
        mean=np.zeros((1,len(success_rates[0])))
        checkpoints=np.zeros((1,len(success_rates[0])))
        for i in range(len(success_rates[0])):
            values=[success_rates[0][i],success_rates[1][i],success_rates[2][i],success_rates[3][i],success_rates[4][i]]
            mean[0][i]=sum(values)/len(values)
            std[0][i]=statistics.pstdev(values)
            checkpoints[0][i]=(checkpoints_interval/max_steps)*(i+1)
        
        plt.plot(checkpoints[0,:],mean[0,:], 'k-',color="blue")
        plt.fill_between(checkpoints[0,:], (mean-std)[0,:], (mean+std)[0,:])
        plt.savefig(address+"/success_rates.png")

    
    
    if destinations is not None:
        # plot coordination of terminal states 
        _, ax = plt.subplots(1, 1, figsize=(5, 4))
        for x, y in env.maze._walls:
            ax.plot(x, y, 'k-')
        

        for destination in destinations:
            
            sample=destination[0]
            
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
    # agent.learn(max_frames,callback=CustomCallback(address=address))
    # agent.save(address+"/sac_agent")

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
    parser.add_argument('-t','--task',required=True)
    args = parser.parse_args()
    
    if args.task=="train":
        main(args.address)
    elif args.task=="plot":
        # plot success rates of all 5 agents
        success_rates=[]
        for i in range(num_agents):
            with open(args.address+"/agent "+str(i+1)+"/success_rates", 'rb') as fp:
                success_rates.append(pickle.load(fp))
        plot(args.address,success_rates=success_rates)

    else:
        pass