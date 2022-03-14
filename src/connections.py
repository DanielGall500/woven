from woven.encoder import WOVEncoder
import numpy as np
import random

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
