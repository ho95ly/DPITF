__author__ = 'Administrator'
import os,sys
import numpy as np
import csv
import matplotlib.pyplot as plt
from function_real import *
from test_sample import *
# from dataprocessor import *
from mpi4py import MPI


comm = MPI.COMM_WORLD
comm_rank = comm.rank
comm_size = comm.size
status = MPI.Status()
root = 0
FINISH = False

<<<<<<< HEAD
sample_number = 30000
filename = 'data/qq.txt'
=======

sample_number = 50000
filename = 'data/1.txt'
>>>>>>> b95a6b8e197d362a8f4cd011187e2108bdb0ab90

if comm_rank == 0:  # master
    num_workers = 3
    closed_workers = 0
    interation_number = 0
    #tensor = np.random.rand(100, 150, 200)*10
    sample_list, e1 = samplefromfile(filename, sample_number)
    np.savetxt('data/e1.txt', e1,fmt='%.1f', delimiter='')
    #shape = tensor.shape
    # shape = get_rowdata_shape(filename)
<<<<<<< HEAD
    shape = (6000, 300, 12)
    print('shape:',shape)
    rank = 10
    delt = 0.9 * sample_number/np.sqrt(shape[0] * shape[1] * shape[2])
=======
    shape = (7000, 200, 12)
    print('shape:',shape)
    rank = 10
    delt = 3 * sample_number/np.sqrt(shape[0] * shape[1] * shape[2])
>>>>>>> b95a6b8e197d362a8f4cd011187e2108bdb0ab90
    tao = 20 * np.sqrt(shape[0] * shape[1] * shape[2])
    # tao = 1700
    epsilon = 10e-3
    tmpe = 0
    y = np.zeros(sample_number)
    #a, b, c = sample3D_rule(shape, sample_number)
    #sample_list = [a, b, c]
    #e1 = Pomega_tensor(sample_list, tensor,  sample_number)
    #print(type(e1))
    #print(type(e1[0]))
    #e1_l2 = np.sqrt(np.add.reduce(e1**2, axis=0))
    e1_l2 = np.linalg.norm(e1, ord=2)
    print("e1 norm:", np.linalg.norm(e1, ord=2))
    # (tensor, tensor_shape, r,delt,tao, epsilon=10e-3, sample_number=0, iteration_num=50)
    para = {"tensor_shape": shape, "rank": rank, "delt": delt, "tao": tao, "epsilon": 10e-3,
                     "sample_number": sample_number, "iteration_num": interation_number, "sample_list": sample_list}
    print("Start the Master.")
    # elist = []
    para = comm.bcast(para, root=0)
    print('bcast')
    while interation_number <= 10000:
        y = comm.bcast(y, root=0)
        X = comm.recv(source=1)
        #print('pomega(X)  recv.')
        Y = comm.recv(source=2)
        #print('pomega(Y)  recv.')
        Z = comm.recv(source=3)
        #print('pomega(Z)  recv.')
        e2 = Pomega_Pair(sample_list, X, Y, Z, shape, sample_number)
        e = e1-e2
        RMSE = 0
        for i in e:
            RMSE += i**2
        RMSE = np.sqrt(RMSE/sample_number)
        # erate = np.sqrt(np.add.reduce(e**2, axis=0))/e1_l2
        erate = np.linalg.norm(e, ord=2)/e1_l2
        print("%d erate is %.8f, RMSE %.5f:" % (interation_number, erate, RMSE))
        #  elist.append(e2)
        #if erate <= epsilon:
        if(erate<=para['epsilon'] or (np.abs(tmpe-erate)<10e-5 and interation_number>1500)):
            print("Master finishing ,erate is :", erate)
            # FW.write(elist)
            FINISH = True
            FINISH = comm.bcast(FINISH, root=0)
            np.savetxt('data/erate.txt', e2 ,fmt='%.1f', delimiter='')
            #f = open('data/result.csv', 'w')
            #f.write(e1)
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
            #print("1 adjoint done.")
            X = shrink(a, para['tao'], mode='complicated')
            #print("1 shrink done.")
            X = X / np.sqrt(para['tensor_shape'][2])
            comm.send(X, dest=0)
        if comm_rank == 2:
            a = adjoint_operator(para['sample_list'], y, para['tensor_shape'],  para['sample_number'], 1)
            #print("2 adjoint done.")
            Y = shrink(a, para['tao'], mode='normal')
            #print("2 shrink done.")
            Y = Y / np.sqrt(para['tensor_shape'][0])
            comm.send(Y, dest=0)
        if comm_rank == 3:
            a = adjoint_operator(para['sample_list'], y, para['tensor_shape'],  para['sample_number'], 2)
            #print('3 adjoint done')
            Z = shrink(a, para['tao'], mode='normal')
            #print('3 shrink done.')
            Z = Z / np.sqrt(para['tensor_shape'][1])
            comm.send(Z, dest=0)

'''
# mple_list = comm.bcast(sample_list, root=0)
# print("test:rank:%d, y:"%comm_rank, y)
# print("sample_list", sample_list)
'''

