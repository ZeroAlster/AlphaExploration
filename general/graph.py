import numpy as np
import math
import random
import sys

class Graph:
    def __init__(self, cell_side,env_start,env_side):
        self.cell_side = cell_side
        self.env_start=env_start
        self.transitions={}
        self.num_cells=env_side/cell_side
        self.max=80

    
    def find(self,coordination): 
        
        cell=self.pos(coordination)

        if cell in self.transitions:
            keys=list(self.transitions[cell].keys())
            index=random.randrange(0,len(keys))
            return self.transitions[cell][keys[index]],np.array(keys[index])
        else:
            return None,None
    
    def add_transition(self,coordination,action,new_coordination):
        
        cell=self.pos(coordination)
        action=(action[0],action[1])

        if cell in self.transitions:
            if action not in self.transitions[cell]:
                if len(self.transitions[cell])<self.max:
                    self.transitions[cell][action]=new_coordination
                else:
                    keys=list(self.transitions[cell].keys())
                    index=random.randrange(0,len(keys))
                    del self.transitions[cell][keys[index]]
                    self.transitions[cell][action]=new_coordination
        else:
            self.transitions[cell]={}
            self.transitions[cell][action]=new_coordination
    

    def pos(self,coordination):
        
        # define the cell
        cell_x=math.floor((coordination[0]-self.env_start[0])/self.cell_side)
        cell_y=math.floor((coordination[1]-self.env_start[1])/self.cell_side)
        cell_x=min(cell_x,self.num_cells-1)
        cell_x=max(cell_x,0)
        cell_y=min(cell_y,self.num_cells-1)
        cell_y=max(cell_y,0)

        return (cell_x,cell_y)


