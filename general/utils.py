import numpy as np
import pickle
import math
import argparse
import matplotlib.pyplot as plt
import statistics
from scipy.interpolate import make_interp_spline
from matplotlib.font_manager import FontProperties
import matplotlib.colors as cor
import pylab
import pandas as pd


# plot individual study for the paper
#####################################
def plot_individual(num_agents,num_methods,environment):
    all_success=[]
    for _ in range(num_methods):
         all_success.append([])

    address1="general/ablation study/exploration/"+environment
    address2="general/ablation study/buffer/"+environment
    address3="general/ablation study/update/"+environment
    address4="DDPG/results/"+environment

    addresses=[]
    addresses.append(address1)
    addresses.append(address2)
    addresses.append(address3)
    addresses.append(address4)

    for j in range(num_methods):
        if j==3:
            num_agents=5
        elif j==1:
             num_agents=3
        else:
             num_agents=3
        for i in range(num_agents):
            with open(addresses[j]+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
                all_success[j].append(pickle.load(fp))


    plt.figure()
    ax = plt.subplot(111)
    colors=["blue","red","green","purple","aqua"]

    for k in range(num_methods):
        number=len(all_success[1][1])
        
        std=[]
        mean=[]
        horizon=[]

        i=0
        beta=1
        while i < number:
            values=[]

            if k==3:
                num_agents=5
            elif k==1:
                num_agents=3
            else:
                num_agents=3

            for j in range(num_agents):
                values.append(all_success[k][j][i])
            mean.append(sum(values)/len(values))
            std.append(statistics.pstdev(values))
            horizon.append(i/beta)

            i+=beta

        mean=np.array(mean)
        std=np.array(std)
        horizon=np.array(horizon)
        
        #smoothing the plots
        window_size = 15

        first_part=mean[0:window_size]
        first_part=pd.Series(first_part).rolling(window=5).mean()
        first_part[0]=0
        first_part[1]=0
        first_part[2]=0
        first_part[3]=0
        first_part[4]=0
        
        mean = pd.Series(mean).rolling(window=window_size).mean()
        mean[0:window_size]=first_part
        plt.plot(horizon, mean,color=colors[k])

        # fix the error bar
        window_size=10
        std=pd.Series(std).rolling(window=window_size).mean()
        std=std*0.4
        down_bar=np.maximum((mean-std),0)
        up_bar=np.minimum((mean+std),1)

        ax.fill_between(horizon,down_bar,up_bar,color=colors[k],alpha=0.1)

    fontP = FontProperties()
    fontP.set_size('x-small')

    plt.title("success rate")
    plt.xlabel("checkpoints")
    plt.savefig(environment)


# plot success rate for the paper
#####################################
def plot_success(num_agents,num_methods,environment):
    all_success=[]
    for _ in range(num_methods):
         all_success.append([])

    address1="our_method/results/"+environment
    address2="DOIE/"+environment
    # address2="our_method/results/"+environment+"/full (replay buffer)"
    address3="PG-baselines/SAC/"+environment
    address4="PG-baselines/TD3/"+environment
    address5="DDPG/results/"+environment

    addresses=[]
    addresses.append(address1)
    addresses.append(address2)
    addresses.append(address3)
    addresses.append(address4)
    addresses.append(address5)

    for j in range(num_methods):
        if j==1:
            for i in range(num_agents):
                with open(addresses[j]+"/seed_"+str(i), 'rb') as fp:
                    all_success[j].append(pickle.load(fp))
        else:
            for i in range(num_agents):
                with open(addresses[j]+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
                    all_success[j].append(pickle.load(fp))


    plt.figure()
    ax = plt.subplot(111)
    colors=["blue","red","green","purple","aqua"]

    for k in range(num_methods):
        number=len(all_success[0][1])
        
        std=[]
        mean=[]
        horizon=[]

        i=0
        beta=1
        while i < number:
            values=[]
            for j in range(num_agents):
                values.append(all_success[k][j][i])
            mean.append(sum(values)/len(values))
            std.append(statistics.pstdev(values))
            horizon.append(i/beta)

            i+=beta

        mean=np.array(mean)
        std=np.array(std)
        horizon=np.array(horizon)
        
        #smoothing the plots
        window_size = 15

        # first_part=mean[0:window_size]
        # first_part=pd.Series(first_part).rolling(window=5).mean()
        # first_part[4]=0
        # first_part[3]=0
        
        mean = pd.Series(mean).rolling(window=window_size).mean()
        # mean[0:window_size]=first_part
        plt.plot(horizon, mean,color=colors[k])

        # fix the error bar
        window_size=10
        std=pd.Series(std).rolling(window=window_size).mean()
        std=std*0.4
        down_bar=np.maximum((mean-std),0)
        up_bar=np.minimum((mean+std),1)

        ax.fill_between(horizon,down_bar,up_bar,color=colors[k],alpha=0.1)

    fontP = FontProperties()
    fontP.set_size('x-small')

    plt.title("success rate")
    plt.xlabel("checkpoints")
    # plt.savefig("general/final_figures/"+environment+"/success rates")
    plt.savefig("test")

# plot environment coverage for the paper
#####################################
def plot_coverage(num_agents,num_methods):
    explorations1=[]
    explorations2=[]
    explorations3=[]
    explorations4=[]
    explorations5=[]
    address1="DDPG_ICM/results/push"
    address2="DDPG_temporal/results/push/full"
    address3="DDPG/results/push"
    address4="our_method/results/push/full (perfect model)"
    address5="DOIE/push/exploration/"

    for i in range(num_agents):
        with open(address1+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
                explorations1.append(pickle.load(fp))
    for i in range(num_agents):
        with open(address2+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
                explorations2.append(pickle.load(fp))
    for i in range(num_agents):
        with open(address3+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
                explorations3.append(pickle.load(fp))
    for i in range(num_agents):
        with open(address4+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
                explorations4.append(pickle.load(fp))
    for i in range(num_agents):
        with open(address5+"/env_coverage_"+str(i+1), 'rb') as fp:
                explorations5.append(pickle.load(fp))

    plt.figure()
    ax = plt.subplot(111)
    explorations=[]
    labels=["DDPG + intrinsic motivation","DDPG + \u03B5z-Greedy ","DDPG","DDPG + \u03B5t-greedy", "DOIE"]
    colors=["aqua","blue","red","green","purple","gold","aqua"]
    explorations.append(explorations1)
    explorations.append(explorations2)
    explorations.append(explorations3)
    explorations.append(explorations4)
    explorations.append(explorations5)

    for k in range(num_methods):
        number=600
        std=np.zeros((1,number))
        mean=np.zeros((1,number))
        horizon=np.zeros((1,number))
        for i in range(number):
            values=[]
            for j in range(num_agents):
                # for mujoco
                # values.append(explorations[k][j][i]*(25/17))
                # for maze
                # values.append(explorations[k][j][i])
                # for push
                values.append(explorations[k][j][i]*(1.09))
            mean[0][i]=sum(values)/len(values)
            std[0][i]=statistics.pstdev(values)
            horizon[0][i]=i
        
        plt.plot(horizon[0,:],mean[0,:],color=colors[k],label=labels[k])

        # fix the error bar
        down_bar=np.maximum((mean-std)[0,:],0)
        up_bar=np.minimum((mean+std)[0,:],1)

        plt.fill_between(horizon[0,:],down_bar,up_bar,color=colors[k],alpha=0.15)

    plt.title("environment coverage")
    plt.xlabel("checkpoints")
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.gca().set_ylim(top=1.05)
    # ax.legend(loc="lower right")
    plt.savefig("test")


# plot a legend for each plot
######################################
def plot_legend():
     
    colors=["blue","red","green","purple","aqua"]
    fig = pylab.figure()
    figlegend = pylab.figure(figsize=(13,0.5))
    ax = fig.add_subplot(111)
    lines = ax.plot(range(10), colors[0], range(10), colors[1], range(10), colors[2], range(10), colors[3], range(10), colors[4])
    figlegend.legend(lines, ("DDPG + \u03B5t-greedy","DDPG + GDRB","DDPG + longest n-step return","DDPG"),loc='center',ncol=4)
    fig.show()
    figlegend.show()
    figlegend.savefig('legend.png',bbox_inches='tight')

# plot the comparison between model and buffer
def plot_model_buffer():
    all_success=[]
    for _ in range(2):
         all_success.append([])

    address1="our_method/results/push/update/avg8-step-TD/"
    address2="our_method/results/push/full (perfect model)/"

    addresses=[]
    addresses.append(address1)
    addresses.append(address2)

    for i in range(5):
        with open(addresses[0]+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
            all_success[0].append(pickle.load(fp))
    
    for i in range(5):
        with open(addresses[1]+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
            all_success[1].append(pickle.load(fp))


    plt.figure()
    ax = plt.subplot(111)
    colors=["blue","red","green","brown"]


    for k in range(2):
        number=len(all_success[0][1])

        std=[]
        mean=[]
        horizon=[]

        i=0
        beta=1
        while i < number:
            values=[]

            num_agents=5
            if(k==0):
                num_agents=5

            for j in range(num_agents):
                values.append(all_success[k][j][i])
            mean.append(sum(values)/len(values))
            std.append(statistics.pstdev(values))
            horizon.append(i/beta)

            i+=beta

        mean=np.array(mean)
        std=np.array(std)
        horizon=np.array(horizon)
        
        #smoothing the plots
        window_size = 15

        first_part=mean[0:window_size]
        first_part=pd.Series(first_part).rolling(window=5).mean()
        first_part[4]=0
        first_part[3]=0
        
        mean = pd.Series(mean).rolling(window=window_size).mean()
        if k==0:
            mean[0:window_size]=first_part
        plt.plot(horizon, mean,color=colors[k])

        # fix the error bar
        window_size=10
        std=pd.Series(std).rolling(window=window_size).mean()
        std=std*0.4
        down_bar=np.maximum((mean-std),0)
        up_bar=np.minimum((mean+std),1)

        ax.fill_between(horizon,down_bar,up_bar,color=colors[k],alpha=0.1)

    fontP = FontProperties()
    fontP.set_size('x-small')

    plt.title("success rate")
    plt.xlabel("checkpoints")
    plt.savefig("test")
    # plt.savefig("general/final_figures/"+"/success rates")


# print number of successes for each agent
######################################
def num_success(num_agents,address,env):    
    if env=="maze":
        goal=[8.8503,9.1610]
        threshold=0.15
    elif env=="point":
        goal=[0,16]
        threshold=0.6
    elif env=="push":
        goal=[4,24.8]
        threshold=0.6
    
    num_agents
    for  i in range(num_agents):
        with open(address+"/agent"+str(i+1)+"/locations", 'rb') as fp:
                    locations=pickle.load(fp)
        success=0
        for k in range(len(locations)):
            location=locations[k][0]
            if math.sqrt(math.pow(location[0]-goal[0],2)+math.pow(location[1]-goal[1],2))<=threshold:
                success+=1
        print("agent"+str(i+1)+":   "+str(success))
        print("frames agent"+str(i+1)+":   "+str(len(locations)))
        print("*"*30)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--address')
    parser.add_argument('-e','--environment')
    parser.add_argument('-m','--method')
    parser.add_argument('-agents',help='number of agents')
    parser.add_argument('-curves',help='number of methods to plot')
    args = parser.parse_args()

    
    # num_success(int(args.agents),args.address,args.environment)
    # plot_coverage(int(args.agents),int(args.curves))
    # plot_legend()
    # plot_success(int(args.agents),int(args.curves),args.environment)
    # plot_model_buffer()
    plot_individual(int(args.agents),int(args.curves),args.environment)