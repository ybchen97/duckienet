import os
import sys
import numpy as np 

fn = sys.argv[1]

with open(fn,'r') as file:
    lines = file.readlines()[1:]
    filtered_sample = list()
    for line in lines:
        tmp = line.split(" ")
        intent = tmp[4][:-1]
        if intent == 'forward':
            filtered_sample.append(str.join(" ",tmp))
    file.close()

with open('filtered.txt','w') as file:
    file.write('frame intention_type current_velocity steering_wheel_angle dlm\n')
    for sample in filtered_sample:
        file.write(sample+'\n')    
    