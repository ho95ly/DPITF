import os,sys
import tensorflow as tf
import numpy as np
import csv
from function import *

tensor_shape = (10, 20, 30)
Tensor = np.random.rand(10, 20, 30)*10
tao = 10*np.sqrt(tensor_shape[0]*tensor_shape[1]*tensor_shape[2])
sample_number = 100
# delt = np.eye(sample_number)
delt = 0.9*sample_number/np.sqrt(tensor_shape[0]*tensor_shape[1]*tensor_shape[2])

def exact_recovery(tensor, tao, delt, ra, rb, rc, epsilon=10e-3, sample_number=0, iteration_num=50):
    y = np.zeros(sample_number)
    # X, Y, Z = 0, 0, 0
    X = np.zeros((tensor_shape[0],tensor_shape[1]))
    Y = np.zeros((tensor_shape[1], tensor_shape[2]))
    Z = np.zeros((tensor_shape[2], tensor_shape[0]))
    a, b, c = sample3D_rule(tensor.shape, sample_number)
    # print('a', a)
    # print('b', b)
    # print('c', c)
    for i in range(iteration_num):

        X, ra = shrinkageA(adjoint_operator(a,b,c, y, tensor.shape, sample_number, 0), tao,ra)
        Y, rb = shrinkageBorC(adjoint_operator(a,b,c, y, tensor.shape, sample_number, 1), tao, rb)
        Z, rc = shrinkageBorC(adjoint_operator(a,b,c, y, tensor.shape, sample_number, 2), tao, rc)

        X = X/np.sqrt(tensor.shape[2])
        Y = Y/np.sqrt(tensor.shape[0])
        Z = Z/np.sqrt(tensor.shape[1])

        e1 = Pomega_tensor(a,b,c, tensor, tensor.shape,sample_number)
        e2 = Pomega_Pair(a,b,c, X, Y, Z, tensor.shape, sample_number)
        e = e1-e2
        if (np.linalg.norm(e, ord=2)/np.linalg.norm(Pomega_tensor(a, b, c, tensor, tensor.shape, sample_number), ord=2) <= epsilon):
            print('########################################')
            break

        y = y + delt*e
        print('interation:',i)

    # matlist = [X, Y, Z]

    # return matlist
    return X, Y, Z


def stable_recovery(Tensor_hat, tao, delt, ra, rb, rc, epsilon, epsilon1, sample_number=0, iteration_num=50):
    y = np.zeros(sample_number)
    s = 0
    X = np.zeros((tensor_shape[0], tensor_shape[1]))
    Y = np.zeros((tensor_shape[1], tensor_shape[2]))
    Z = np.zeros((tensor_shape[2], tensor_shape[0]))
    a, b, c = sample3D_rule(tensor.shape, sample_number)
    # print('a', a)
    # print('b', b)
    # print('c', c)
    for i in range(iteration_num):
        X, ra = shrinkageA(adjoint_operator(a, b, c, y, Tensor_hat.shape, sample_number, 0), tao, ra)
        Y, rb = shrinkageBorC(adjoint_operator(a, b, c, y, Tensor_hat.shape, sample_number, 1), tao, rb)
        Z, rc = shrinkageBorC(adjoint_operator(a, b, c, y, Tensor_hat.shape, sample_number, 2), tao, rc)
        X = X / np.sqrt(Tensor_hat.shape[2])
        Y = Y / np.sqrt(Tensor_hat.shape[0])
        Z = Z / np.sqrt(Tensor_hat.shape[1])

        e = Pomega_tensor(a, b, c, Tensor_hat,Tensor_hat.shape, sample_number)-Pomega_Pair(a, b, c, X, Y, Z,Tensor_hat.shape, sample_number)  # how to compute the elementwise difference between Pomega(Tensor_hat) and Pomega(Pair(...)) ?
        if np.linalg.norm(e, ord=2) / np.linalg.norm(Pomega_tensor(a, b, c, Tensor_hat,Tensor_hat.shape, sample_number), ord=2) <= epsilon:
            print('########################################')
            break

        y = y + delt * e
        s = s - delt * epsilon1
        y, s = cone_projection_operator(y, s)
    # matlist = [X, Y, Z]
    return X, Y, Z

qq,qw,qe = exact_recovery(Tensor, tao, delt, 5, 5, 5, 10e-3, 1000, 100)

print('X', qq)
print('Y', qw)
print('Z', qe)
'''
aa,ad,af = stable_recovery(Tensor, tao, delt, 5, 5, 5, 10e-3, 10e-3, 100, 1000)
print('X', aa)
print('Y', ad)
print('Z', af)

'''



