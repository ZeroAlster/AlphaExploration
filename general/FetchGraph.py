from simhash import Simhash
import random
import numpy as np

class Model:
    def __init__(self,env,item_dim=3,dim_key=12,bucket_size=5000):
        self.transitions={}
        self.densities={}
        self.env=env
        self.projection_matrix = np.random.normal(size=(item_dim, dim_key))

        
        # precompute modulos of powers of 2
        mods_list = []
        mod = 1
        mods = []
        for _ in range(dim_key):
            mods.append(mod)
            mod = (mod * 2) % bucket_size
        mods_list.append(mods)
        self.mods_list = np.asarray(mods_list).T
        self.bucket_size=bucket_size
    
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
            self.densities[hash]=1
        self.densities[hash]+=1

    def get_density(self,state):
        hash=self.pos(state)
        if hash not in self.densities:
            return 0
        return self.densities[hash]

    def reset(self):
        self.densities={}

    def pos(self,state):
        coordinates=np.round(state[0:3],3)

        binary=np.sign(coordinates.dot(self.projection_matrix))
        keys = np.cast['int'](binary.dot(self.mods_list)) % self.bucket_size

        return keys[0]