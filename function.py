import os,sys
import tensorflow as tf
import numpy as np
import csv



tensor = np.random.rand(8, 4, 5)*10
mat = np.random.rand(80, 40)*10
row = mat.shape[0]
col = mat.shape[1]
vecr1 = np.ones((row, 1))
vecr2 = np.ones((col, 1))
delta = np.matmul(np.matmul(np.transpose(vecr1),mat),vecr2)/np.sqrt(row*col)


def generate(shape, rank):

    U = np.random.randn(shape[0], rank)
    V = np.random.randn(shape[1], rank)
    return U, V


def centralization(mat):
    tmp = np.matmul(np.ones((mat.shape[0], mat.shape[0])), mat)/mat.shape[0]
    centermat = mat-tmp
    return centermat


def subspace(shape, rank, mode=None):
    U, V = generate(shape, rank)
    tmp = np.matmul(U, np.transpose(V))
    if mode == 'B':
        Psb = centralization(tmp)
        return Psb
    if mode == 'C':
        Psc = centralization(tmp)
        return Psc
    elif mode == 'A':
        row = tmp.shape[0]
        col = tmp.shape[1]
        vec1 = np.ones((row, 1))
        vec1_t = np.transpose(vec1)
        vec2 = np.ones((col, 1))
        vec2_t = np.transpose(vec2)
        Psa = centralization(tmp) + np.matmul(np.matmul(vec1_t, tmp), vec2) * (vec1 * vec2_t) / (row * col)
        '''
        ss = centralization(tmp)
        a = np.matmul(vec1_t, tmp)
        b = np.matmul(a, vec2)
        c = b * (vec1 * vec2_t)
        d = c / (row * col)
        e = ss+d
        return e
        '''
        return Psa
    return False


def sample_rule4mat(tensor_shape,ra, rb, rc, sample_num):
    A = subspace((tensor_shape[0], tensor_shape[1]), ra, 'A')
    B = subspace((tensor_shape[1], tensor_shape[2]), rb, 'B')
    C = subspace((tensor_shape[2], tensor_shape[0]), rc, 'C')
    return A, B, C


def sample3D_rule(tensor_shape, sample_num):
    len_1d = tensor_shape[0]
    len_2d = tensor_shape[1]
    len_3d = tensor_shape[2]
    a = np.random.randint(0, len_1d, sample_num)
    b = np.random.randint(0, len_2d, sample_num)
    c = np.random.randint(0, len_3d, sample_num)
    return a, b, c


def Pomega_mat(a,b,c, mat, tensor_shape, m, dim=None):
    # delta_vec = np.eye(m, dtype=int)
    #sum_vec = 0
    sum_vec = []
    # a, b, c = sample3D_rule(tensor_shape, m)
    if dim == 0:
        X = mat
        for i in range(m):
            # sum_vec += X[a[i]][b[i]] * delta_vec[:, i]
            sum_vec.append(X[a[i]][b[i]])
        sum_vec = np.asarray(sum_vec)
        Pomega_AX = sum_vec/np.sqrt(tensor_shape[2])
        return Pomega_AX
    elif dim == 1:
        Y = mat
        for i in range(m):
            # sum_vec += Y[b[i]][c[i]] * delta_vec[:, i]
            sum_vec.append(Y[b[i]][c[i]])
        sum_vec = np.asarray(sum_vec)
        Pomega_BY = sum_vec/np.sqrt(tensor_shape[0])
        return Pomega_BY
    elif dim == 2:
        Z = mat
        for i in range(m):
            #sum_vec += Z[c[i]][a[i]] * delta_vec[:, i]
            sum_vec.append(Z[c[i]][a[i]])
        sum_vec = np.asarray(sum_vec)
        Pomega_CZ = sum_vec/ np.sqrt(tensor_shape[1])
        return Pomega_CZ


def adjoint_operator(a, b, c, sample_vec, tensor_shape,m ,dim=None):# sample_vec[i]????  OR sample_vec[i][0]???
    # a, b, c = sample3D_rule(tensor_shape, m)
    if dim == 0:
        # sample_vec = Pomega_mat(a, b, c, mat, tensor_shape, m, 0)
        X = np.zeros((tensor_shape[0], tensor_shape[1]))
        for i in range(m):
            X[a[i]][b[i]] = sample_vec[i]
        return X
    elif dim == 1:
        # sample_vec = Pomega_mat(a, b, c, mat, tensor_shape, m, 1)
        Y = np.zeros((tensor_shape[1], tensor_shape[2]))
        for i in range(m):
            Y[b[i]][c[i]] = sample_vec[i]
        return Y
    elif dim == 2:
        # sample_vec = Pomega_mat(a, b, c, mat, tensor_shape, m, 2)
        Z = np.zeros((tensor_shape[2], tensor_shape[0]))
        for i in range(m):
            Z[c[i]][a[i]] = sample_vec[i]
        return Z
    else:
        print('error')
        return -1


def Pomega_tensor(a,b,c, tensor, tensor_shape, sample_number):
    # a, b, c = sample3D_rule(tensor_shape, sample_number)
    # sample_t = np.zeros((sample_number, 1))
    sample_t = np.zeros(sample_number)
    for i in range(sample_number):
        sample_t[i] = tensor[a[i]][b[i]][c[i]]
    # print(sample_t)
    return sample_t


def Pomega_Pair(a,b,c, X, Y, Z, tensor_shape, m):
    Pomega_A = Pomega_mat(a,b,c, X, tensor_shape, m, 0)
    Pomega_B = Pomega_mat(a,b,c, Y, tensor_shape, m, 1)
    Pomega_C = Pomega_mat(a,b,c, Z, tensor_shape, m, 2)
    Pomega_pair = Pomega_A + Pomega_B + Pomega_C
    # print(Pomega_pair)
    return Pomega_pair


def cone_projection_operator(x, t):
    norm_x = np.linalg.norm(x, ord=2)
    if norm_x <= t:
        return x, t
    if t <= np.negative(norm_x):
        return 0, 0
    if t >= np.negative(norm_x) and t <= norm_x:
        tmp = (norm_x + t)/(2 * norm_x)
        return tmp*x, tmp*norm_x


def SVT(mat, tao):
    u,s,v = np.linalg.svd(mat, full_matrices=True, compute_uv=True)
    for i in range(len(s)):
        s[i]=max(0,s[i]-tao)
    # print(u, s, v)
    return u,s,v

'''
def shrink(mat, mode='normal'):
        u,s,v = SVT(mat)
        if mode == 'normal':
            return np.matmul(np.matmul(u,s),v)
        if mode == 'complicated':
            delta = np.matmul(np.matmul(np.transpose(vecr1), mat),vecr2)/np.sqrt(row*col)
            tmp1 = np.matmul(np.matmul(u,s),v)
            tmp2 = (max(0,delta-tao)+min(0,delta+tao))*np.ones((mat.shape))/np.sqrt(row*col)
            return tmp1+tmp2
'''
def shrink(mat, tao, mode='normal'):
        u,s,v = SVT(mat, tao)
        sm = np.zeros((u.shape[0], v.shape[0]))
        for i in range(len(s)):
            sm[i][i] = s[i]
        if mode == 'normal':
            return np.matmul(np.matmul(u, sm), v)
        if mode == 'complicated':
            vecr1 = np.ones((mat.shape[0], 1))
            vecr2 = np.ones((mat.shape[1], 1))
            delta = np.matmul(np.matmul(np.transpose(vecr1), mat),vecr2)/np.sqrt(mat.shape[0]*mat.shape[1])
            tmp1 = np.matmul(np.matmul(u,sm),v)
            tmp2 = (max(0,delta-tao)+min(0,delta+tao))*np.ones((mat.shape))/np.sqrt(mat.shape[0]*mat.shape[1])
            return tmp1+tmp2



def shrinkageBorC(X_hat, tao, r):
    sum = 0
    s = r + 1
    U, S, V = np.linalg.svd(centralization(X_hat))
    '''
    while True:
        if (s + 5 < len(S)):
            s = s + 5
            if (S[s-5] <= tao):
                break
        else:
            if (S[s] <= tao):
                break
            s = s + 1
            if(s>=len(S)):

    for j in range(s-5, s):
        if(S[j] > tao):
            r = j
                 break
    '''
    for i in range(s, len(S)):
        if(S[i]<tao):
            r = i-1
            break


    for j in range(r):
        shape1 = U.shape
        shape2 = V.shape
        m = np.matmul(np.reshape(U[j,:],(shape1[0], 1)), np.reshape(np.transpose(V[j, :]), (1,shape2[0])))
        sum = sum +(S[j]-tao)*m
    X = sum
    return X, r


def shrinkageA(X_hat, tao, r):
    X, r = shrinkageBorC(X_hat, tao, r)
    # delta = np.inner(np.ones(X_hat.shape), X_hat) # elementwise sum of X_hat
    delta = np.sum(X_hat)
    gamma = (max(0, delta-tao)+min(0, delta+tao))/(X_hat.shape[0]*X_hat.shape[1])
    ones_vec1 = np.reshape(np.ones(X_hat.shape[0]), (X_hat.shape[0], 1))
    ones_vec2 = np.reshape(np.ones(X_hat.shape[1]), (1, X_hat.shape[1]))
    result = X + gamma*np.matmul(ones_vec1, ones_vec2)

    return result, r
