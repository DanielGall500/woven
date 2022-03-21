from woven.encoder import WOVEncoder
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

"""
Pseudocode
For each row (the English words) r:
    Assign a colour to r

Flatten matrix into (,cols)
    where matrix[i] = the row it's true in

Combine the above to assign a colour to all outputs
"""
def gen_hex_colour():
    r = lambda: random.randint(50,200)
    return '#%02X%02X%02X' % (r(),r(),r())

def make_colour_connections(encoding):
    nrows = encoding.shape[0]
    ncols = encoding.shape[1]
    colours = [gen_hex_colour() for i in range(nrows)]
    
    hex_default = '#FFFFFF'
    output_colours = []

    for i, row in enumerate(encoding.T):
        #Note: iterates the cols of encoding
        found_true = np.where(row == 1)[0]
        num_found_true = len(found_true)
        if num_found_true == 0:
            output_colours.append(hex_default)
        elif num_found_true == 1:
            appropriate_colour = colours[found_true[0]]
            output_colours.append(appropriate_colour)
        else:
            #Note: Information lost due to limit of colouring
            appropriate_colour = colours[found_true[0]]
            output_colours.append(appropriate_colour)
            #raise Exception(encoding)
    return colours, output_colours

def create_graph(itokens, otokens, icolours, ocolours, encoding):
    G = nx.Graph() 
    total_tokens = len(itokens) + len(otokens)

    curr_x = 0
    x = [x for x in range(0,2000,int(2000/total_tokens))]
    for i, itok in enumerate(itokens):
        G.add_node(int(i), label=itok, physics=False,x=x[i], y=200, color=icolours[i]) 

    total_nodes = G.number_of_nodes()
    for i, otok in enumerate(otokens):
        next_node_indx = int(i+total_nodes)
        G.add_node(next_node_indx, label=otok, physics=False, x=x[i], y=100, color=ocolours[i])

    default_edges_input = [(i,i+1) for i in range(len(itokens)-1)]
    default_edges_output = [(i+total_nodes,i+total_nodes+1) for i in range(len(otokens)-1)]
    G.add_edges_from(default_edges_input,color=icolours)
    G.add_edges_from(default_edges_output,color=ocolours)

    for i,(a,b) in enumerate(default_edges_input):
        G.add_edge(a,b,color='black')

    for i,(a,b) in enumerate(default_edges_output):
        G.add_edge(a,b,color='black')

    for i, row in enumerate(encoding):
        #Get the word for each input row
        connections = np.where(row == 1)[0]
        connections = [int(x) for x in connections]

        for c in connections:
            #Adds an edge from input to output
            G.add_edge(i,c+total_nodes,color='red')
    nt = Network('800px', '1000px')
    nt.from_nx(G)
    return nt
