import numpy as np

# Generate major key templates
major_template = np.array([[1,0,1,0,1,1,0,1,0,1,0,1]])
# Generate monor key templates
minor_template = np.array([[1,0,1,1,0,1,0,1,1,0,1,0]])
template = np.empty((0,12))

#generate 2-D array with every shifts
for i in range(12):
    template = np.append(template, np.roll(major_template, i), axis=0)
for i in range(12):
    template = np.append(template, np.roll(minor_template, i), axis=0)






