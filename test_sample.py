__author__ = 'Administrator'
import os,sys
import numpy as np
import csv
import matplotlib.pyplot as plt
from function_real import *
# from dataprocessor import *
from mpi4py import MPI

def deal_timestamp(arr_file, col):
    shape = arr_file.shape
    time = arr_file[:, col]
    max_time = max(time)
    min_time = min(time)
    timesum = 0
    for row in range(shape[0]):
       timesum += time[row]
    timeavg = timesum/shape[0]
    k = (max_time-min_time)//shape[0]
    for row in range(shape[0]):
       time[row] = (time[row]-timeavg)/k
    print("deal time success with liner method.")
    return time

def get_rowdata_shape(filename):
    shapelist = []
    my_matrix = np.loadtxt(filename)
    # deal_timestamp(my_matrix, 3)
    for i in range(3):
        shapelist.append(int(max(my_matrix[:, i])))
    tensorshape = tuple(shapelist)
    return tensorshape

def get_data_shape(sample_list):
     shapelist=[]
     for i in range(3):
         shapelist.append(max(arr[i]))
     tensorshape = tuple(shapelist)
     return tensorshape

def samplefromfile(filename, sp_num):
    # filename = 'data/1.txt'
    f = open(filename, 'r')
    a = []
    b = []
    c = []
    v = []
    while sp_num > 0:
        item = f.readline()
        item = item.split()
        a.append(int(item[0])%7000)
        b.append(int(item[1])%200)
        c.append(int(item[2])%12)
        v.append(float(item[3]))
        sp_num -= 1
    sample_list = [a, b, c]
    value = v
    return sample_list, value
