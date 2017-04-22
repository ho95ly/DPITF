import os,sys
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from function import *

def exact_recovery(tensor, tensor_shape, r,delt,tao, epsilon=10e-3, sample_number=0, iteration_num=50):
    y = np.zeros(sample_number)
    tmpe = 0
    # X, Y, Z = 0, 0, 0
    #X = np.zeros((tensor_shape[0],tensor_shape[1]))
    #Y = np.zeros((tensor_shape[1], tensor_shape[2]))
    #Z = np.zeros((tensor_shape[2], tensor_shape[0]))
    Ef = []
    If = []

    a, b, c = sample3D_rule(tensor.shape, sample_number)
    # A, B, C = sample_rule4mat(tensor_shape, r, r, r, sample)
    #delt = 0.9 * sample_number/ np.sqrt(tensor_shape[0] * tensor_shape[1] * tensor_shape[2])
    #tao = 10 * np.sqrt(tensor_shape[0] * tensor_shape[1] * tensor_shape[2])
    fd = r * (tensor_shape[0] + tensor_shape[1]-r) + r * (tensor_shape[1] + tensor_shape[2] - r) + r * (tensor_shape[2] + tensor_shape[0]-r)
    print('degree of freedom:', fd)
    print('delt:',delt)
    print('m/d:', sample_number/fd)

    for i in range(iteration_num):

        X = shrink(adjoint_operator(a,b,c, y, tensor.shape, sample_number, 0), tao, mode='complicated')
        Y = shrink(adjoint_operator(a,b,c, y, tensor.shape, sample_number, 1), tao, mode='normal')
        Z = shrink(adjoint_operator(a,b,c, y, tensor.shape, sample_number, 2), tao, mode='normal')

        X = X/np.sqrt(tensor.shape[2])
        Y = Y/np.sqrt(tensor.shape[0])
        Z = Z/np.sqrt(tensor.shape[1])

        e1 = Pomega_tensor(a,b,c, tensor, tensor.shape,sample_number)
        # e1 = Pomega_Pair(a,b,c, A, B, C, tensor_shape, sample_number)
        e2 = Pomega_Pair(a,b,c, X, Y, Z, tensor.shape, sample_number)
        e = e1-e2

        erate = np.linalg.norm(e, ord=2)/np.linalg.norm(e1, ord=2)
        # erate = np.linalg.norm(e, ord=2) / np.linalg.norm(e1, ord=2)


        Ef.append(erate)
        If.append(i)

        if (erate <= epsilon):
            print('########################################')
            print(erate)
            break
        if (np.abs(tmpe - erate) < 10e-6 and i >1000):
            break
        y = y + delt*e
        tmpe = erate
        print('interation:', i, erate)

    # matlist = [X, Y, Z]
    # return matlist

    return [X, Y, Z], If, Ef



Tensor = np.random.rand(20, 30, 40)*10
Tensor1 = np.random.rand(50, 70,80)*10
Tensor2  = np.random.rand(1000,50,8)*10
# tensor_shape = Tensor.shape
sample= 20000
interation = 10000
Ef = []  # error list
If = []  # interation list
plt.figure(figsize=(10, 10))


'''
for i in range(1,4):
    qq, qw, qe = exact_recovery(Tensor, 20, 10e-3, 3000*i, 50)
    plt.plot(If, Ef, linewidth=2,label='line')
    Ef = []
    If = []
'''
tensor_shape = (100,150,200)
delt = 0.9 * 8000/ np.sqrt(tensor_shape[0] * tensor_shape[1] * tensor_shape[2])
tao = 10 * np.sqrt(tensor_shape[0] * tensor_shape[1] * tensor_shape[2])
delt1 = 0.7 * 8000/ np.sqrt(tensor_shape[0] * tensor_shape[1] * tensor_shape[2])
tao1 = 10 * np.sqrt(tensor_shape[0] * tensor_shape[1] * tensor_shape[2])
delt2 = 0.6 * 8000/ np.sqrt(tensor_shape[0] * tensor_shape[1] * tensor_shape[2])
tao2 = 10 * np.sqrt(tensor_shape[0] * tensor_shape[1] * tensor_shape[2])



q,w,e = exact_recovery(Tensor2, (1000,50,8), 5,delt,tao, 10e-3, 8000, interation)
q1,w1,e1 = exact_recovery(Tensor2, (1000,50,8), 5, delt1,tao1,10e-3, 8000, interation)
q2,w2,e2 = exact_recovery(Tensor2, (1000,50,8), 5,delt2,tao2,10e-3, 8000, interation)
#q3,w3,e3 = exact_recovery(Tensor, 10,10e-3, 8000, interation)
# q3,w3,e3 = exact_recovery(Tensor, 20, 10e-3, 3000*i, interation)
plt.plot(w, e, linewidth=2, label='delt 0.9')
plt.plot(w1, e1, linewidth=2, label='delt 0.7')
plt.plot(w2, e2, linewidth=2, label='delt 0.6 ')
# plt.plot(w3, e3, linewidth=2, label='rank 50')
plt.xlabel("interation")
plt.ylabel("error rate")
plt.legend()
plt.show()

