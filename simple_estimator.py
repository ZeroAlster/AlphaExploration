import numpy as np
import math


minimum_prob=0.001

class SEstimator:
    def __init__(self, cell_side,env_xside,env_yside,env_start):
        self.cell_side = cell_side
        self.env_start=env_start
        self.visits=np.zeros((math.ceil(env_xside/cell_side),math.ceil(env_yside/cell_side)))+0.001
    
    def prob(self,coordination):
        cell_x=math.floor((coordination[0]-self.env_start[0])/self.cell_side)
        cell_y=math.floor((coordination[1]-self.env_start[1])/self.cell_side)

        cell_x=min(cell_x,self.visits.shape[0]-1)
        cell_x=max(cell_x,0)
        cell_y=min(cell_y,self.visits.shape[1]-1)
        cell_y=max(cell_y,0)
        
        prob=(self.visits[cell_x][cell_y])/np.sum(self.visits)
        return prob

    def increment(self,coordination):
        cell_x=math.floor((coordination[0]-self.env_start[0])/self.cell_side)
        cell_y=math.floor((coordination[1]-self.env_start[1])/self.cell_side)
        
        cell_x=min(cell_x,self.visits.shape[0]-1)
        cell_x=max(cell_x,0)
        cell_y=min(cell_y,self.visits.shape[1]-1)
        cell_y=max(cell_y,0)

        self.visits[cell_x][cell_y]+=1

