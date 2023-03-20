import gym
from agent import Agent
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import math
import pickle
from collections import deque
from env import Env

alpha=0.4


def plot(rewards):
    plt.plot(rewards)
    plt.savefig("test.png")


def main():
    env = gym.make("Pendulum-v0")
    agent=Agent(1,3,2)
    rewards=[]
    total_step = 0
    for i in range(10000):
        total_reward = 0
        step =0
        state = env.reset()
        for t in count():
            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state,float(done))

            state = next_state
            if done:
                break
            step += 1
            total_reward += reward
        
        total_step += step+1
        rewards.append(total_reward)
        if i%1000==0:
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
        for k in range(5):
            agent.update(min(200,len(agent.memory)))
    
    plot(rewards)


if __name__ == '__main__':
    #main()
    print(0.5*math.pow(0.9999994,int(4e6)))