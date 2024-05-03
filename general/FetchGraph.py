from simhash import Simhash
import random
import sys

class Model:
    def __init__(self,env):
        self.transitions={}
        self.densities={}
        self.env=env
        self.granularity=self.set_granularity()


    def set_granularity(self):
        if self.env=="reach":
            return 16
        else:
            return 16
    
    def find(self,state):         
        hash=self.pos(state)

        if hash in self.transitions:
            response=random.choice(self.transitions[hash])
            return response[0],response[1]
        else:
            return None,None
    
    def add_transition(self,state,action,next_state):
        hash=self.pos(state)
        if hash not  in self.transitions:
            self.transitions[hash]=[]
        self.transitions[hash].append((next_state,action))
    
    def increment(self,state):
        hash=self.pos(state)
        if hash not in self.densities:
            self.densities[hash]=0
        self.densities[hash]+=1

    def get_density(self,state):
        hash=self.pos(state)
        if hash not in self.densities:
            return 0
        return self.densities[hash]

    def reset(self):
        self.densities={}

    def pos(self,state):
        # get the coordinates
        if self.env=="reach":
            coordinates=state[0:3]
        elif self.env=="push" or self.env=="slide":
            coordinates=state[0:6] 
        else:
            raise TypeError("the environment is not allowed.")

        coord_str = ",".join(map(str, coordinates))
        simhash_value = Simhash(coord_str,f=self.granularity)
        simhash_hex_str = simhash_value.value
        return simhash_hex_str


