"""
visual image and its intention
"""
import cv2
import time 
import getch 
import os
import math
import numpy as np

LABEL_PATH = '/media/duong/Data10/duong/data/interpolated_data.txt'
BASE = '/media/duong/Data10/duong/data/denoise_data'
VALID_KEYS = ['a','l','r','x','n']
SLIDE_MAX = 100

frame = list()
dlm = list()
# read recorded data
with open(LABEL_PATH,'r') as file:
    lines = file.readlines()[1:]
    for line in lines:
        tmp = line.split(" ")
        frame.append(tmp[0])
        dlm.append(tmp[4][:-1]) #remove \n

def display(i):
    fn = os.path.join(BASE,frame[i]+'.jpg')
    im = cv2.imread(fn)
    cv2.putText(im,str(i),bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
    cv2.putText(im,dlm[i], (80,100),font,fontScale,(0,255,0),lineType)
    return im

i = 0 
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,200)
fontScale = 0.5
fontColor = (255,255,255)
lineType = 2
char = None

while True:
    print('enter a key')
    if char is None:
        char = getch.getch()
    if char == 'x':
        print('exit...')
        break
    elif char == 'n':
        i = int(input())
        im = display(i)
        cv2.imshow('im',im)
        char = str(chr(cv2.waitKey(0))) 
        if char in VALID_KEYS:
            cv2.destroyAllWindows()
            continue
    elif char == 'a':
        slide = True 
        while True:
            i = (i+1)%len(dlm)
            im = display(i)
            cv2.imshow('im',im)
            char = cv2.waitKey(40) 
            if char != -1:
                char = str(chr(char))
                if char in VALID_KEYS and char != 'a':
                    cv2.destroyAllWindows()
                    break
    else:
        if char == 'l':
            i = (i+1)%len(dlm)
            im = display(i)
            cv2.imshow('im',im)
            char = str(chr(cv2.waitKey(0))) 
            if char in VALID_KEYS:
                cv2.destroyAllWindows()
                continue
        elif char == 'r':
            i = (i-1)%len(dlm)
            im = display(i)
            cv2.imshow('im',im)
            char = str(chr(cv2.waitKey(0))) #pauses for 3 seconds before fetching next image
            if char in VALID_KEYS:
                cv2.destroyAllWindows()
                continue
        else:
            print('invalid char: ',char,"please type in 'a','l','r','x','n'")
            char = None
            
