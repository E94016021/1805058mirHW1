import numpy as np

major_template = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]])
minor_template = np.array([[6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])

kstemplate = np.empty((0, 12))

for i in range(12):
    kstemplate = np.append(kstemplate, np.roll(major_template, i), axis=0)
for i in range(12):
    kstemplate = np.append(kstemplate, np.roll(minor_template, i), axis=0)
