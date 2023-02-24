import pickle

# write list to binary file
def write_list(a_list):
    # store list in binary file so 'wb' mode
    with open('listfile', 'wb') as fp:
        pickle.dump(names, fp)
        print('Done writing list into a binary file')

# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('listfile', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

# list of names
names = ['Jessa', 'Eric', 'Bob']
write_list(names)
r_names = read_list()
print('List is', r_names[1])