import numpy as np
import math
import random
import sys

class Model:
    def __init__(self, x_bounds, y_bounds, z_bounds, cell_size):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.cell_size = cell_size

        self.x_cells = int((x_bounds[1] - x_bounds[0]) / cell_size)
        self.y_cells = int((y_bounds[1] - y_bounds[0]) / cell_size)
        self.z_cells = int((z_bounds[1] - z_bounds[0]) / cell_size)
        
        self.grid = np.zeros((self.x_cells, self.y_cells, self.z_cells), dtype=int)

    
    def increment(self, state):
        x_index, y_index, z_index = self.pos(state)
        self.grid[x_index, y_index, z_index] += 1
    
    def get_density(self, state):
        x_index, y_index, z_index = self.pos(state)
        return self.grid[x_index, y_index, z_index]

    def pos(self, state):
        x=state[0]
        y=state[1]
        z=state[2]
        x_index = int((x - self.x_bounds[0]) / self.cell_size)
        y_index = int((y - self.y_bounds[0]) / self.cell_size)
        z_index = int((z - self.z_bounds[0]) / self.cell_size)
        
        # Ensure indices are within bounds
        x_index = np.clip(x_index, 0, self.x_cells - 1)
        y_index = np.clip(y_index, 0, self.y_cells - 1)
        z_index = np.clip(z_index, 0, self.z_cells - 1)

        return x_index, y_index, z_index


