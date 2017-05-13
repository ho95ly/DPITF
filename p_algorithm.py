__author__ = 'Administrator'

import os,sys
import numpy as np
import csv
import matplotlib.pyplot as plt
from function import *
from mpi4py import MPI


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)
tags = enum('READY', 'DONE', 'EXIT', 'START')


# Define MPI message tags


comm = MPI.COMM_WORLD
comm_rank = comm.rank
comm_size = comm.size
status = MPI.Status()
root = 0
FW2 = open('result.csv', 'w')
FW1 = open('true.csv', 'w')

FINISH = False
if comm_rank == 0:  # master
    num_workers = 3
    closed_workers = 0
    interation_number = 0
    tensor = np.random.rand(100, 150, 200)*10
    sample_number = 16800
    shape = tensor.shape
    rank = 20
    delt = 0.8 * sample_number/np.sqrt(shape[0] * shape[1] * shape[2])
    tao =  15* np.sqrt(shape[0] * shape[1] * shape[2])
    epsilon = 10e-3
    y = np.ones(sample_number)
    a, b, c = sample3D_rule(shape, sample_number)
    sample_list = [a, b, c]
    e1 = Pomega_tensor(sample_list, tensor, shape, sample_number)
    e1_l2 = np.sqrt(np.add.reduce(e1**2, axis=0))
    fd = 2*rank*(shape[0]+shape[1]+shape[2])-3*rank**2
    print('degree of freedom:',fd)
    print("e1 norm:", np.linalg.norm(e1, ord=2))
    # (tensor, tensor_shape, r,delt,tao, epsilon=10e-3, sample_number=0, iteration_num=50)
    para = {"tensor_shape": shape, "rank": rank, "delt": delt, "tao": tao, "epsilon": 10e-3,
                     "sample_number": sample_number, "iteration_num": interation_number, "sample_list": sample_list}
    print("Start the Master.")
    elist = []
    tmpe = 0 
    para = comm.bcast(para, root=0)
    while interation_number <= 10000:
        y = comm.bcast(y, root=0)
        X = comm.recv(source=1)
        # print('pomega(X)  recv.')
        Y = comm.recv(source=2)
        # print('pomega(Y)  recv.')
        Z = comm.recv(source=3)
        # print('pomega(Z)  recv.')
        e2 = Pomega_Pair(sample_list, X, Y, Z, tensor.shape, sample_number)
        e = e1-e2
        erate = np.sqrt(np.add.reduce(e**2, axis=0))/e1_l2
        # erate = np.linalg.norm(e, ord=2)/np.linalg.norm(e1, ord=2)
        print("%d erate is %.8f:" % (interation_number, erate))
        if(erate<=para['epsilon'] or (np.abs(tmpe-erate)<10e-5 and interation_number>2000)):       
            print("Master finishing ,erate is :", erate)
            for i in range(sample_number):
                FW2.write('%f\n'%e2[i])
                FW1.write('%f\n'%e1[i])
            FW1.close()    
            FW2.close()
            FINISH = True
            FINISH = comm.bcast(FINISH, root=0)
            # f = open('data/result.csv', 'w')
            # f.write(sample_list)
            # writer = csv.writer(f)
            # writer.writerows(sample_list)
            break
        
        interation_number += 1
        y += delt * e
        tmpe = erate
    
    print("Master finishing ERROR!")
else:  # slave
    para = comm.bcast(None, root=0)
    # print("test:rank:%d, para:" % comm_rank, para)
    while True:
        y = comm.bcast(None, root=0)
        if FINISH:
            print("%d has done." % comm_rank)
            break
        if comm_rank == 1:
            a = adjoint_operator(para['sample_list'], y, para['tensor_shape'],  para['sample_number'], 0)
            X = shrink(a, para['tao'], mode='complicated')
            X = X / np.sqrt(para['tensor_shape'][2])
            comm.send(X, dest=0)
        if comm_rank == 2:
            a = adjoint_operator(para['sample_list'], y, para['tensor_shape'],  para['sample_number'], 1)
            Y = shrink(a, para['tao'], mode='normal')
            Y = Y / np.sqrt(para['tensor_shape'][0])
            comm.send(Y, dest=0)
        if comm_rank == 3:
            a = adjoint_operator(para['sample_list'], y, para['tensor_shape'],  para['sample_number'], 2)
            Z = shrink(a, para['tao'], mode='normal')
            Z = Z / np.sqrt(para['tensor_shape'][1])
            comm.send(Z, dest=0)

'''
# mple_list = comm.bcast(sample_list, root=0)
# print("test:rank:%d, y:"%comm_rank, y)
# print("sample_list", sample_list)
'''
