import pickle
from general.env import Env
import matplotlib.pyplot as plt
from random import sample







with open("DDPG_temporal/results/agent6/locations", 'rb') as fp:
    locations=pickle.load(fp)

locations=sample(locations,4000)



# plot locations on the map
env=Env(n=100,maze_type='square_large')
_, ax = plt.subplots(1, 1, figsize=(5, 4))
for x, y in env.maze._walls:
    ax.plot(x, y, 'k-')

# remove the outliers
locations_x=[]
locations_y=[]
sampler=[]
for destination in locations:

    # we plot the results until 6M frames
    if destination[1]>250000:
        continue
    
    sample=destination[0][0:2]
    
    if sample[0]>9.5 or sample[1]>9.5:
        #sample=[min(sample[0],9.5),min(sample[1],9.5)]
        continue
    if sample[0]<-0.5 or sample[1]<-0.5:
        #sample=[max(sample[0],-0.5),max(sample[1],-0.5)]
        continue

        
    locations_x.append(sample[0])
    locations_y.append(sample[1])

# plot the locations
ax.scatter(locations_x,locations_y,marker='o',c="red",s=10)
plt.savefig("destinations_temporal.png")
plt.close("all")