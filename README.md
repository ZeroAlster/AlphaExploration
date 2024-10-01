This is the code for ETGL-DDPG project. 



Install required libraries:

pip install -r requirements.txt

DOIE Results:

The results for the DOIE algorithm are generated using their publicly available code.

Plot Generation:

All data needed to generate the plots in the paper is provided in the results directory of each algorithm, with random seeds ranging from 1 to 100.


Running the Algorithms:

To run an algorithm and generate results:

* Navigate to the respective algorithm's directory.
* Execute the main.py file with the appropriate arguments. 

Maze Environments:

U-maze and Point-push are built using the mujoco-maze package, which is included in the installed libraries. You can customize these environments by modifying mujoco-maze/maze_task.py after installation.