import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import pandas as pd

# Should pass in the data as this:: python A1_KNN.py Data4A1_NoDup.tsv 5 10

no_of_observation = 400
file_name = sys.argv[1] # tab-delimited plain-text file name.
k = int(sys.argv[2]) # the number of closest neighbours (k) in integer.
unknown_instance_number = int(sys.argv[3]) # the number of unknown instances in integer

df = pd.read_csv(file_name, sep='\t', header=0) # read the text file
## check validity of k and U
# check if k is in range, if not, print error message
if k <= 1 and k >= (no_of_observation - unknown_instance_number):
    print('k should be greater or equal to 1, and less than or equal to # of Observation minus U. \n please re-enter your 2nd argument')
## check if U is in range, if not, print error message
elif unknown_instance_number > no_of_observation:
    print('U should be less than the observation number. \n please re-enter your 3rd argument')

# Randomly select U instances to be the unknown instances in UnInstance
unknownInstance = df.sample(unknown_instance_number)

# Remove the unknown instances in UnInstance from the input data. The remaining instances are the training data T.
T = df.drop(index=unknownInstance.index, axis=0)

# calculate the distance with the chi squared distanca Formula


def chiDistance(x, y):
    distance = 0.5 * np.sum([((x - y) ** 2) / (x + y)
                             for (x, y) in zip(x, y)])
    return distance
