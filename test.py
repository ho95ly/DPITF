import numpy as np

tensor_shape = (8, 4, 5)
tao = 10*np.sqrt(tensor_shape[0]*tensor_shape[1]*tensor_shape[2])
# tensor = np.random.rand(8, 4, 5)*10
mat = np.random.rand(80, 40)*10
row = mat.shape[0]
col = mat.shape[1]
vecr1 = np.ones((row, 1))
vecr2 = np.ones((col, 1))
delta = np.matmul(np.matmul(np.transpose(vecr1),mat),vecr2)/np.sqrt(row*col)
tensor = np.random.rand(5,5,5)*10
print(tensor)

def centralization(mat):
    tmp = np.matmul(np.ones((row,row)), mat)/row
    centermat = mat-tmp
    return centermat


def shrinkageBorC(X_hat, tao, r):
    sum = 0
    s = r + 1
    while True:
        U, S, V = np.linalg.svd(centralization(X_hat))
        # Stmp = S[0:s] # return top s singular vectors of S
        if (s + 5 < len(S)):
            s = s + 5
        else:
            s = len(S)-1

        if(s-5 >= 0):
            if(S[s-5]<=tao):
                break

    for j in range(s-5, s):
        if(S[j] > tao):
            r = j
    print(r)

    return r
