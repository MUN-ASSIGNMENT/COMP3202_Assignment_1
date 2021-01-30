import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import pandas as pd

no_of_observation = 400
# Should pass in the data as this:: python3 A1_KNN.py data.txt 5 15 

# tab-delimited plain-text file name.
file_name = sys.argv[1]
# the number of closest neighbours (k) in integer.
k = int(sys.argv[2])
# the number of unknown instances in integer
unknown_instance_number = int(sys.argv[3])
print("file name: ", file_name,"\nnumber of closest neighbours: ", k, "\nnumber of unknown instances", unknown_instance_number)
df = pd.read_csv(file_name, sep='\t', header=0)

## check validity of k and U
# check if k is in range, if not, print error message
if k <= 1 and k >= (no_of_observation - unknown_instance_number): 
    print('k should be greater or equal to 1, and less than or equal to # of Observation minus U. \n please re-enter your 2nd argument')
## check if U is in range, if not, print error message
elif unknown_instance_number > no_of_observation: 
    print('U should be less than the observation number. \n please re-enter your 3rd argument')

# Randomly select U instances to be the unknown instances in UnInstance 
unknownInstance = df.sample(unknown_instance_number)
print(unknownInstance)
column = unknownInstance.iloc[:, 0]
print(column)
print(df)
df.drop(column)
print(df)

# 4. Remove the unknown instances in UnInstance from the input data. The remaining
# instances are the training data T.
# Calculate distance between two points
# Function to calculate Chi-distace 


# for all the unknown instances UnInstance(i)
# for all the known instances in training data T(j)
# compute the distance between UnInstance(i) and
# T(j)
# end for
# find the k smallest distances
def chiDistance(x, y): 
    # calculate the distance with the chi squared distanca Formula
    distance = 0.5 * np.sum([((x - y) ** 2) / (x + y)  
        for (x, y) in zip(x, y)]) 
  
    return distance 
  
# main function 
if __name__== "__main__": 
    a = [1, 2, 13, 5, 45, 23] 
    b = [67, 90, 18, 79, 24, 98] 
  
    result = chiDistance(a, b) 
    print("The Chi-square distance is :", result) 