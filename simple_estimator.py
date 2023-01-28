import numpy as np
import math

class SEstimator:
    def __init__(self, cell_side,env_xside,env_yside,env_start):
        self.cell_side = cell_side
        self.env_start=env_start
        self.visits=np.zeros((math.ceil(env_xside/cell_side),math.ceil(env_yside/cell_side)))
    
    def prob(self,coordination):
        cell_x=math.ceil((coordination[0]-self.env_start[0])/self.cell_side)
        cell_y=math.ceil((coordination[1]-self.env_start[1])/self.cell_side)
        prob=self.visits[cell_x][cell_y]/np.sum(self.visits)
        return prob

    def increment(self,coordination):
        cell_x=math.ceil((coordination[0]-self.env_start[0])/self.cell_side)
        cell_y=math.ceil((coordination[1]-self.env_start[1])/self.cell_side)
        self.visits[cell_x][cell_y]+=1

