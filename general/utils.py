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


# plot success rate for the paper
#####################################
def plot_success(num_agents,num_methods,environment):
    all_success=[]
    for _ in range(num_methods):
         all_success.append([])

    address1="DDPG/results/"+environment
    address2="our_method/results/"+environment
    address3="PG+HER/SAC/"+environment
    address4="PG+HER/TD3/"+environment

    addresses=[]
    addresses.append(address1)
    addresses.append(address2)
    addresses.append(address3)
    addresses.append(address4)
         

    for j in range(num_methods):
        for i in range(num_agents):
            with open(addresses[j]+"/agent"+str(i+1)+"/success_rates", 'rb') as fp:
                all_success[j].append(pickle.load(fp))


    plt.figure()
    ax = plt.subplot(111)
    colors=["blue","red","green","brown"]


    for k in range(num_methods):
        number=len(all_success[k][0])

        std=[]
        mean=[]
        horizon=[]

        i=0
        beta=10
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
        X_Y_Spline = make_interp_spline(horizon, mean)
        X_ = np.linspace(horizon.min(), horizon.max(),50)
        Y_ = X_Y_Spline(X_)
        Y_=np.minimum(Y_,1)
        Y_=np.maximum(Y_,0)
        plt.plot(X_, Y_,color=colors[k])

        # fix the error bar
        std=std*0.6
        down_bar=np.maximum((mean-std),0)
        up_bar=np.minimum((mean+std),1)

        ax.fill_between(horizon,down_bar,up_bar,color=colors[k],alpha=0.1)

    fontP = FontProperties()
    fontP.set_size('x-small')

    plt.title("success rate")
    plt.xlabel("checkpoints")
    plt.savefig("general/final_figures/"+environment+"/success rates")


# plot environment coverage for the paper
#####################################
def plot_coverage():
    explorations1=[]
    explorations2=[]
    explorations3=[]
    explorations4=[]
    explorations5=[]
    address1="DDPG_ICM/results/push"
    address2="DDPG_temporal/results/push/full"
    address3="DDPG/results/push"
    address4="our_method/results/push/full (perfect model)"
    address5="our_method/results/push/full (replay buffer)"

    for i in range(10):
        with open(address1+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
                explorations1.append(pickle.load(fp))
    for i in range(10):
        with open(address2+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
                explorations2.append(pickle.load(fp))
    for i in range(10):
        with open(address3+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
                explorations3.append(pickle.load(fp))
    for i in range(10):
        with open(address4+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
                explorations4.append(pickle.load(fp))
    for i in range(10):
        with open(address5+"/agent"+str(i+1)+"/env_coverage", 'rb') as fp:
                explorations5.append(pickle.load(fp))

    plt.figure()
    ax = plt.subplot(111)
    explorations=[]
    labels=["DDPG + intrinsic motivation","DDPG + \u03B5z-Greedy ","DDPG","DDPG + \u03B5t-greedy (perfect model)", "DDPG + \u03B5t-greedy (replay buffer)"]
    colors=["darkorange","blue","red","green","purple","gold","aqua"]
    explorations.append(explorations1)
    explorations.append(explorations2)
    explorations.append(explorations3)
    explorations.append(explorations4)
    explorations.append(explorations5)

    for k in range(5):
        number=600
        std=np.zeros((1,number))
        mean=np.zeros((1,number))
        horizon=np.zeros((1,number))
        for i in range(number):
            values=[]
            agents=10
            # if k==0:
            #     agents=5
            # elif k==1:
            #      agents=5
            # elif k==4:
            #      agents=12 

            for j in range(agents):
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

        plt.fill_between(horizon[0,:],down_bar,up_bar,color=colors[k],alpha=0.2)

    plt.title("environment coverage")
    plt.xlabel("checkpoints")
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.gca().set_ylim(top=1.05)
    # ax.legend(loc="lower right")
    plt.savefig("test")


# plot a legend for each plot
######################################
def plot_legend():
     
    colors=["darkorange","blue","red","green","purple","saddlebrown","aqua"]
    fig = pylab.figure()
    figlegend = pylab.figure(figsize=(13,0.5))
    ax = fig.add_subplot(111)
    lines = ax.plot(range(10), colors[0], range(10), colors[1], range(10), colors[2], range(10), colors[3], range(10), colors[4])
    figlegend.legend(lines, ("DDPG","DDPG + \u03B5t-Greedy (perfect model)","DDPG + \u03B5t-Greedy (replay buffer)",
                             "DDPG + GDRB","DDPG + longest ns-tep return"),loc='center',ncol=5)
    fig.show()
    figlegend.show()
    figlegend.savefig('legend.png',bbox_inches='tight')


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
    # plot_coverage()
    # plot_legend()
    plot_success(int(args.agents),int(args.curves),args.environment)